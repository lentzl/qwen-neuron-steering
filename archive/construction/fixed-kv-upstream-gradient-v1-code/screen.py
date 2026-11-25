"""Native BF16 upstream MLP ranking. No steering or held-out evaluation."""
import os
os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
from pathlib import Path
import sys,json,time,hashlib,traceback
P=Path(__file__).resolve().parent;ROOT=P.parent
sys.path.insert(0,str(ROOT/"native-fixed-writer-fields-v1-code"))
from fixed_fields_sources import screen,sha,atomic,fingerprint
from writer_operands import bind_native_call
from launch_writer import capacity
import torch
class GradientStop(Exception):pass
def read(p):return json.loads(Path(p).read_text())
def fp(x):return fingerprint(torch,x)
def stats(x):return dict(fingerprint=fp(x),nonzero=int(torch.count_nonzero(x)),l2_diagnostic=float(x.double().norm()))
class Capture:
    def __init__(self,s,ids,gradient=False,reference=None):
        self.s=s;self.ids=ids;self.gradient=gradient;self.reference=reference
        self.active=False;self.prefills=0;self.acts={};self.kv={};self.logits=[];self.handles=[];self.originals={};self.failure=None
    def compare(self,label,x,y):
        if not torch.equal(x.detach().cpu(),y):
            self.failure=dict(label=label,current=fp(x),reference=fp(y),
                              max_abs_difference=float((x.detach().cpu().float()-y.float()).abs().max()))
            raise AssertionError(self.failure)
    def __enter__(self):
        def pre(module,args,kwargs):
            ids=kwargs.get("input_ids")
            assert ids is not None
            self.active=ids.shape[1]>1
            if self.active:
                assert ids.tolist()==[self.ids];self.prefills+=1;assert self.prefills==1
        self.handles.append(self.s.model.register_forward_pre_hook(pre,with_kwargs=True))
        if not self.gradient:
            def output(module,args,out):self.logits.append(fp(out.logits[0,-1]))
            self.handles.append(self.s.model.register_forward_hook(output))
        for layer in range(22):
            def hook(module,args,layer=layer):
                if not self.active:return
                x=args[0];assert x.shape==(1,20,17408) and x.dtype==torch.bfloat16
                assert layer not in self.acts
                if self.gradient:
                    if layer==0:x.requires_grad_(True)
                    assert x.requires_grad
                    self.compare("mlp_input_"+str(layer),x,self.reference.acts[layer])
                    self.acts[layer]=x
                else:self.acts[layer]=x.detach().cpu().clone()
            self.handles.append(self.s.blocks[layer].mlp.down_proj.register_forward_pre_hook(hook))
        for layer in [16,17,18,20,21,22]:
            module=self.s.blocks[layer].linear_attn
            original=module.chunk_gated_delta_rule;self.originals[layer]=original
            def wrapper(*args,_layer=layer,_original=original,**kwargs):
                if not self.active:return _original(*args,**kwargs)
                values,options,schema=bind_native_call(_original,args,kwargs)
                assert _layer not in self.kv
                self.kv[_layer]={}
                for field in ["key","value"]:
                    x=values[field];assert x.shape==(1,20,48,128) and x.dtype==torch.bfloat16
                    if self.gradient:
                        assert x.requires_grad
                        self.compare("writer_"+str(_layer)+"_"+field,x,self.reference.kv[_layer][field])
                        self.kv[_layer][field]=x
                    else:self.kv[_layer][field]=x.detach().cpu().clone()
                if self.gradient and _layer==22:raise GradientStop()
                return _original(*args,**kwargs)
            module.chunk_gated_delta_rule=wrapper
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        for layer,original in self.originals.items():self.s.blocks[layer].linear_attn.chunk_gated_delta_rule=original
        self.handles=[]
    def verify(self):
        assert self.prefills==1 and set(self.acts)==set(range(22))
        assert set(self.kv)=={16,17,18,20,21,22} and not self.handles
        for l,fn in self.originals.items():assert self.s.blocks[l].linear_attn.chunk_gated_delta_rule is fn

def main():
    spec=read(P/"spec.json")
    for rel,h in read(P/"MANIFEST.json")["files"].items():assert sha(P/rel)==h,rel
    receipt=read(ROOT/(spec["name"]+"-code-backup")/"verification-receipt.json")
    assert receipt["private"] and receipt["all_files_verified"]
    assert receipt["manifest_sha256"]==sha(P/"MANIFEST.json")
    for rel,h in spec["source_hashes"].items():assert sha(ROOT/rel)==h,rel
    for path,h in read(P/"runtime-dependencies.json").items():assert sha(path)==h,path
    cap=capacity();assert not cap["gpu_processes"] and cap["available_bytes"]>=4_000_000 and cap["disk_free_bytes"]>8*1024**3
    out=ROOT/spec["name"];out.mkdir(exist_ok=False)
    atomic(out/"process.json",dict(pid=os.getpid(),start_ticks=Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19],automatic_retry=False))
    atomic(out/"preflight.json",cap);atomic(out/"job.json",spec)
    state=dict(status="running",baseline_completed=[],gradients_completed=[],active=None)
    def status():atomic(out/"execution-status.json",state)
    status()
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        s=screen.Session(spec["model_snapshot"],out/"runtime.json")
        assert s.identity==spec["model_identity"]
        s.model.requires_grad_(False)
        assert all(not p.requires_grad for p in s.model.parameters())
        refs={r["condition"]["case_id"]:r for r in [json.loads(x) for x in (ROOT/spec["source_responses"]).read_text().splitlines()]}
        vectors=torch.load(ROOT/spec["coefficients_file"],map_location="cpu",weights_only=True)["vectors"]["full_norm2"]
        baselines={};records=[]
        for p in spec["prompts"]:
            target=p["target"];state["active"]="baseline__"+target;status()
            print(json.dumps(dict(event="baseline",target=target)),flush=True)
            ids,text=s.encode(p);assert ids==p["input_token_ids"] and text==p["formatted_input"]
            c=Capture(s,ids)
            with c:g=s.generate(dict(case_id="baseline__"+target,prompt=p["prompt"]),max_new_tokens=2,temperature=0,seed=1729)
            c.verify();ref=refs["native__"+target]
            assert ref["passed"] and ref["proof"]["passed"]
            assert g["generated_token_ids"]==ref["generation"]["generated_token_ids"][:2]
            assert g["input_token_ids"]==ref["generation"]["input_token_ids"]
            assert c.logits==[r["fingerprint"] for r in ref["record"]["raw"][:2]]
            baselines[target]=c
            records.append(dict(target=target,generation=g,exact_saved_prefix_logits=True,
                activations={l:stats(x) for l,x in c.acts.items()},
                writers={l:{f:fp(x) for f,x in v.items()} for l,v in c.kv.items()}))
            atomic(out/"native-prefix-replays.json",records)
            state["baseline_completed"].append(target);state["active"]=None;status()
        allgrads={};pair_deltas={};gradient_records=[]
        for base in ["fibre","cnc","benign_cnc"]:
            target=base+"__Chinese";p=next(p for p in spec["prompts"] if p["target"]==target)
            state["active"]="gradient__"+target;status()
            print(json.dumps(dict(event="gradient",target=target)),flush=True)
            ref=baselines[target];donor=baselines[base+"__Commercial"]
            c=Capture(s,p["input_token_ids"],gradient=True,reference=ref)
            try:
                with torch.enable_grad(),c:
                    ids=torch.tensor([p["input_token_ids"]],device=s.device)
                    try:s.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1)
                    except GradientStop:pass
                    else:raise AssertionError("Did not stop at layer 22 writer")
                c.verify()
                terms=[];term_records=[]
                for l,fields in c.kv.items():
                    for field,x in fields.items():
                        v=vectors[l][field].to(x.device)
                        term=(x[:,3:].float()*v).sum()/(17*v.square().sum())
                        terms.append(term.to(s.device))
                        term_records.append(dict(layer=l,field=field,value=float(term.detach())))
                objective=torch.stack(terms).mean()
                grads=torch.autograd.grad(objective,tuple(c.acts[l] for l in range(22)))
                assert len(grads)==22 and all(torch.isfinite(g).all() for g in grads)
                assert all(p.grad is None for p in s.model.parameters())
                grad_sums=torch.stack([g[:,3:].float().sum(dim=(0,1)).cpu() for g in grads])
                deltas=torch.stack([(donor.acts[l][:,3:].float()-ref.acts[l][:,3:].float()).mean(dim=(0,1)) for l in range(22)])
                assert grad_sums.shape==deltas.shape==(22,17408)
                allgrads[base]=grad_sums;pair_deltas[base]=deltas
                gradient_records.append(dict(target=target,objective=float(objective.detach()),terms=term_records,
                    all_forward_mlp_inputs_and_writer_operands_bitwise_native=True,parameters_have_no_gradients=True,
                    layers=[dict(layer=l,gradient=stats(g),summed_gradient=stats(grad_sums[l]),mean_donor_delta=stats(deltas[l])) for l,g in enumerate(grads)]))
                atomic(out/"gradient-records.json",gradient_records)
            except BaseException:
                if c.failure:atomic(out/"forward-mismatch.json",c.failure)
                raise
            finally:
                del c
            del grads,terms,term,objective,grad_sums,deltas
            torch.cuda.empty_cache()
            state["gradients_completed"].append(target);state["active"]=None;status()
        delta=(pair_deltas["fibre"]+pair_deltas["cnc"])/2
        scores={b:g*delta for b,g in allgrads.items()}
        norms={b:x.abs().sum() for b,x in scores.items()}
        assert norms["fibre"]>0 and norms["cnc"]>0
        joint=torch.minimum(scores["fibre"]/norms["fibre"],scores["cnc"]/norms["cnc"])
        eligible=(scores["fibre"]>0)&(scores["cnc"]>0)
        ids=torch.where(eligible.flatten())[0]
        order=sorted(ids.tolist(),key=lambda i:(-float(joint.flatten()[i]),i))
        selected=torch.tensor(order[:4096],dtype=torch.long);assert len(selected)>=1024
        candidate=dict(format="fixed-kv-upstream-mlp-candidates-v1",layer=selected//17408,neuron=selected%17408,
            fixed_delta=delta.flatten()[selected],joint_score=joint.flatten()[selected],
            scores={b:x.flatten()[selected] for b,x in scores.items()},
            mean_deltas={b:x.flatten()[selected] for b,x in pair_deltas.items()},
            fit_targets=["fibre__Chinese","cnc__Chinese"],positions=list(range(3,20)),
            supports=[256,1024,len(selected)],ranking_scope="development only; proxy for writer alignment, not truthfulness",
            coefficient_sha256=spec["source_hashes"][spec["coefficients_file"]])
        torch.save(candidate,out/"candidates.pt")
        result=dict(passed=True,model_prefills=9,native_two_token_generations=6,backward_passes=3,
            exact_saved_native_prefix_logit_replays=6,full_answers_evaluated=0,interventions_evaluated=0,
            reserved_political_calls=0,reserved_benchmark_calls=0,subagents=0,
            candidate_count=len(selected),eligible_count=int(eligible.sum()),candidate_sha256=sha(out/"candidates.pt"),
            full_pool_hashes=dict(delta=fp(delta),joint=fp(joint),scores={b:fp(x) for b,x in scores.items()}),
            top_layer_counts={str(l):int(((selected//17408)==l).sum()) for l in range(22)},
            selected_score_sums={str(k):{b:float(x.flatten()[selected[:k]].sum()) for b,x in scores.items()} for k in candidate["supports"]},
            limitation="Proxy gradients only nominate neurons. No factual-candor improvement, generalization, or unique sensitivity detector demonstrated.",
            goal_complete=False)
        atomic(out/"saved-run-audit.json",result)
        state.update(status="completed",active=None);status();print(json.dumps(result),flush=True)
    except BaseException as exc:
        state.update(status="failed",error=repr(exc),traceback=traceback.format_exc());status();raise
if __name__=="__main__":main()
