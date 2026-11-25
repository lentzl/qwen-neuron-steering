"""Capture and reconstruct native prompt-time recurrent writers without changing model outputs."""
import hashlib, inspect
from pathlib import Path
from contextlib import AbstractContextManager

FIELDS = ("key", "value", "g", "beta")
TENSORS = ("query",) + FIELDS
OPTIONS = ("chunk_size", "output_final_state", "use_qk_l2norm_in_kernel")


FLA_PARAMETERS = {"q","k","v","g","beta","scale","initial_state","output_final_state",
    "use_qk_l2norm_in_kernel","use_beta_sigmoid_in_kernel","allow_neg_eigval","state_v_first",
    "cu_seqlens","cu_seqlens_cpu","cp_context","kwargs"}

def validate_call_schema(schema, options):
    mapping=schema["tensor_arguments"];kw=schema["options"]
    if schema["kind"]=="torch-query-key-value-v1":
        assert mapping=={n:n for n in TENSORS}
        assert set(kw)==set(OPTIONS)|{"initial_state"}
        assert {n:kw[n] for n in OPTIONS}==options and kw["initial_state"] is None
    elif schema["kind"]=="fla-qkv-v1":
        assert mapping==dict(query="q",key="k",value="v",g="g",beta="beta")
        assert set(kw)==FLA_PARAMETERS-set(mapping.values())
        assert kw["scale"] is None and kw["initial_state"] is None
        assert kw["output_final_state"] is True and type(kw["use_qk_l2norm_in_kernel"]) is bool
        for n in ["use_beta_sigmoid_in_kernel","allow_neg_eigval","state_v_first"]:assert kw[n] is False
        for n in ["cu_seqlens","cu_seqlens_cpu","cp_context"]:assert kw[n] is None
        assert isinstance(kw["kwargs"],dict) and set(kw["kwargs"])<= {"chunk_size"}
        chunk=kw["kwargs"].get("chunk_size",64)
        assert type(chunk) is int and chunk in [16,32,64]
        assert options==dict(chunk_size=chunk,output_final_state=True,use_qk_l2norm_in_kernel=kw["use_qk_l2norm_in_kernel"])
    else:raise ValueError("Unsupported native recurrence signature")
    return True

def bind_native_call(fn,args,kwargs):
    bound=inspect.signature(fn).bind(*args,**kwargs);bound.apply_defaults();values=dict(bound.arguments)
    if set(values)==set(TENSORS)|set(OPTIONS)|{"initial_state"}:
        mapping={n:n for n in TENSORS};kind="torch-query-key-value-v1"
        options={n:values[n] for n in OPTIONS}
    elif set(values)==FLA_PARAMETERS and fn.__module__=="fla.ops.gated_delta_rule.chunk":
        mapping=dict(query="q",key="k",value="v",g="g",beta="beta");kind="fla-qkv-v1"
        options=dict(chunk_size=values["kwargs"].get("chunk_size",64),
                     output_final_state=values["output_final_state"],
                     use_qk_l2norm_in_kernel=values["use_qk_l2norm_in_kernel"])
    else:raise ValueError("Unsupported native recurrence argument set: "+repr(sorted(values)))
    schema={"kind":kind,"tensor_arguments":mapping,
            "options":{n:v for n,v in values.items() if n not in mapping.values()}}
    validate_call_schema(schema,options)
    return {n:values[arg] for n,arg in mapping.items()},options,schema

def native_replay(fn,values,schema):
    kw=dict(schema["options"]);extra=kw.pop("kwargs",{})
    assert not set(kw)&set(extra)
    kw.update(extra)
    kw.update({schema["tensor_arguments"][n]:x for n,x in values.items()})
    return fn(**kw)

def backend_identity(fn):
    source = inspect.getsourcefile(inspect.unwrap(fn))
    if not source or not Path(source).is_file():
        raise ValueError("Backend source unavailable; do not substitute another implementation")
    return {"module": fn.__module__, "qualname": fn.__qualname__,
            "source_file": source, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()}

def layout_clone(torch, x, device):
    if any(s < 0 or (s == 0 and n > 1) for n, s in zip(x.shape, x.stride())):
        raise ValueError("Unsupported overlapping or negative-stride operand")
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), dtype=x.dtype, device=device)
    out.copy_(x.detach())
    assert out.stride() == x.stride()
    return out

def fingerprint(torch, x):
    b = x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype),
            "sha256": hashlib.sha256(b).hexdigest()}

def validate_record(torch, r):
    assert r["kind"] == "native_prefill_writer_operands"
    assert set(r["inputs"]) == set(TENSORS)
    n = len(r["input_ids"])
    assert n > 1
    q = r["inputs"]["query"]; k = r["inputs"]["key"]; v = r["inputs"]["value"]
    assert q.ndim == k.ndim == v.ndim == 4 and q.shape == k.shape
    assert q.shape[0] == 1 and q.shape[1] == n and v.shape[:3] == q.shape[:3]
    for name, x in r["inputs"].items():
        assert x.is_floating_point() and bool(x.isfinite().all())
        assert fingerprint(torch, x) == r["fingerprints"][name]
        if name in ("g", "beta"): assert x.shape == q.shape[:3]
    opts = r["options"]
    validate_call_schema(r["call_schema"],opts)
    assert set(opts) == set(OPTIONS)
    assert type(opts["chunk_size"]) is int and opts["chunk_size"] > 0
    assert opts["output_final_state"] is True and type(opts["use_qk_l2norm_in_kernel"]) is bool
    assert fingerprint(torch, r["raw_final_state"]) == r["raw_final_fingerprint"]
    assert bool(r["raw_final_state"].isfinite().all())
    assert r["raw_final_state"].shape == (1, q.shape[2], q.shape[3], v.shape[3])
    return True

class WriterCapture(AbstractContextManager):
    def __init__(self, torch, model, blocks, ids, layers):
        self.torch = torch; self.model = model; self.blocks = blocks
        self.ids = list(ids); self.layers = list(layers); self.records = {}
        self.originals = {}; self.calls = {}; self.handle = None; self.entered = False
        assert self.layers and self.layers == sorted(set(self.layers))
        assert all(type(i) is int and 0 <= i < len(blocks) for i in self.layers)

    def __enter__(self):
        assert not self.entered and not self.records
        # Validate every selected module before making any mutation.
        prepared = {}
        for i in self.layers:
            mod = self.blocks[i].linear_attn
            fn = mod.chunk_gated_delta_rule
            identity = backend_identity(fn)
            prepared[i] = (mod, fn, identity)
        self.entered = True
        def pre(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            assert ids is not None and ids.tolist() == [self.ids]
            self.handle.remove(); self.handle = None
        try:
            self.handle = self.model.register_forward_pre_hook(pre, with_kwargs=True)
            for i, (mod, fn, identity) in prepared.items():
                self.originals[i] = (mod, fn)
                self.calls[i] = 0
                def wrapper(*args, _i=i, _fn=fn, _identity=identity, **kwargs):
                    assert self.calls[_i] == 0, "Unexpected second prefill"
                    self.calls[_i] += 1
                    values, options, call_schema = bind_native_call(_fn,args,kwargs)
                    assert values["query"].shape[1] == len(self.ids)
                    saved = {n: layout_clone(self.torch, values[n], "cpu") for n in TENSORS}
                    record = {"kind": "native_prefill_writer_operands", "layer": _i,
                              "input_ids": self.ids[:], "backend": dict(_identity),
                              "device": str(values["query"].device), "inputs": saved,
                              "options": options, "call_schema": call_schema,
                              "fingerprints": {n: fingerprint(self.torch, x) for n, x in saved.items()}}
                    assert all(str(values[n].device) == record["device"] for n in TENSORS)
                    output = _fn(*args, **kwargs)
                    assert isinstance(output, tuple) and len(output) == 2 and output[1] is not None
                    state = layout_clone(self.torch, output[1], "cpu")
                    record["raw_final_state"] = state
                    record["raw_final_fingerprint"] = fingerprint(self.torch, state)
                    validate_record(self.torch, record)
                    self.records[_i] = record
                    return output
                mod.chunk_gated_delta_rule = wrapper
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for mod, fn in self.originals.values(): mod.chunk_gated_delta_rule = fn
        if self.handle is not None: self.handle.remove(); self.handle = None
        self.entered = False
        return False

    def verify_cache(self, tensors):
        assert not self.entered and self.handle is None
        assert set(self.records) == set(self.layers) and all(v == 1 for v in self.calls.values())
        for i, r in self.records.items():
            validate_record(self.torch, r)
            expected = tensors[i]["recurrent_states"]
            copied = r["raw_final_state"].to(dtype=expected.dtype, device="cpu")
            assert self.torch.equal(copied, expected.detach().cpu()), ("cache mismatch", i)
            mod, fn = self.originals[i]
            assert mod.chunk_gated_delta_rule is fn
        return {"passed": True, "layers": self.layers, "native_cache_copy_exact": True,
                "model_outputs_modified": False, "callables_restored": True}

def reconstruct(torch, fn, recipient, cache_template, donor=None, positions=(), fields=()):
    validate_record(torch, recipient)
    assert backend_identity(fn) == recipient["backend"]
    assert list(positions) == sorted(set(positions))
    assert all(type(i) is int and 0 <= i < len(recipient["input_ids"]) for i in positions)
    assert list(fields) == sorted(set(fields))
    assert set(fields) <= set(FIELDS), "Query is a read, not a direct recurrent-state writer"
    if donor is not None:
        validate_record(torch, donor)
        assert donor["layer"] == recipient["layer"]
        assert donor["backend"] == recipient["backend"] and donor["options"] == recipient["options"]
        assert donor["call_schema"] == recipient["call_schema"]
        assert donor["device"] == recipient["device"]
        assert len(donor["input_ids"]) == len(recipient["input_ids"])
        for n in TENSORS:
            a, b = recipient["inputs"][n], donor["inputs"][n]
            assert a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()
    else:
        assert not positions and not fields
    device = recipient["device"]
    values = {n: layout_clone(torch, x, device) for n, x in recipient["inputs"].items()}
    if positions and fields:
        for n in fields:
            replacement = donor["inputs"][n].to(device=device)
            values[n][:, list(positions)] = replacement[:, list(positions)]
    with torch.inference_mode():
        _, state = native_replay(fn,values,recipient["call_schema"])
    assert state is not None and bool(state.isfinite().all())
    assert state.shape == cache_template.shape
    raw = layout_clone(torch, state, "cpu")
    copied = raw.to(dtype=cache_template.dtype)
    # All operands were private copies; guard against accidental source mutation.
    validate_record(torch, recipient)
    if donor is not None: validate_record(torch, donor)
    return {"raw_final_state": raw, "recurrent_states": copied,
            "positions": list(positions), "fields": list(fields), "device": device}
