"""Prepare fixed, unrefitted layer partitions for later development-only causal tests.
No model loading, inference, CUDA access, reserved-answer reads, or launch.
"""
import argparse,hashlib,json,os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
torch.set_num_threads(1)
EXPECTED="9e82162d138cb8785f5b38e6617c16b6d0d4ec566ce0cceaca3513af2ab7f71e"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open("x") as f: json.dump(x,f,indent=2,sort_keys=True)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    assert sha(a.source)==EXPECTED
    assert not a.output.exists(), "Output must be new; frozen artifacts are immutable"
    packed=torch.load(a.source,map_location="cpu",weights_only=True)
    assert set(packed)=={"mlp","random"}
    variants={}; summaries={}
    for arm in ["mlp","random"]:
        original=packed[arm]
        assert set(original)==set(range(22))
        for layer,v in original.items():
            assert set(v)=={"indices","delta"}
            assert v["indices"].dtype==torch.long and v["delta"].dtype==torch.float32
            assert v["indices"].shape==v["delta"].shape and v["indices"].ndim==1
            assert bool(torch.isfinite(v["delta"]).all())
            assert v["indices"].unique().numel()==v["indices"].numel()
        for name,layers in [("full",set(original)),("layers19_20",{19,20}),("complement",set(original)-{19,20})]:
            values={l:{k:v.clone() for k,v in original[l].items()} for l in sorted(layers)}
            key=arm+"__"+name; variants[key]=values
            summaries[key]=dict(layers=sorted(layers),neurons=sum(v["indices"].numel() for v in values.values()),
               squared_raw_offset_norm=sum(float(v["delta"].double().square().sum()) for v in values.values()))
        focus=variants[arm+"__layers19_20"]; rest=variants[arm+"__complement"]
        assert set(focus).isdisjoint(rest) and set(focus)|set(rest)==set(original)
        for l,v in original.items():
            reconstructed=(focus if l in focus else rest)[l]
            assert all(torch.equal(v[k],reconstructed[k]) for k in v)
        assert summaries[arm+"__full"]["neurons"]==1024
        assert summaries[arm+"__layers19_20"]["neurons"]==132
        assert summaries[arm+"__complement"]["neurons"]==892
    for l in packed["mlp"]:
        m,r=packed["mlp"][l],packed["random"][l]
        assert torch.equal(m["delta"],r["delta"])
        assert set(m["indices"].tolist()).isdisjoint(r["indices"].tolist())
    assert sha(a.source)==EXPECTED
    a.output.mkdir()
    torch.save(dict(variants=variants),a.output/"offset-partitions.pt")
    restored=torch.load(a.output/"offset-partitions.pt",map_location="cpu",weights_only=True)
    for name,values in variants.items():
        assert set(restored["variants"][name])==set(values)
        for l,v in values.items():
            assert all(torch.equal(v[k],restored["variants"][name][l][k]) for k in v)
    audit=dict(passed=True,source=str(a.source),source_sha256=EXPECTED,
       artifact_sha256=sha(a.output/"offset-partitions.pt"),partitions=summaries,
       coefficient_refitting=False,normalization=False,model_loaded=False,inference_calls=0,
       cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],reserved_answers_read=False,
       treatment_maps_read=False,launch_performed=False,
       diagnostics_precision="FP64 norm reductions only; indices int64, offsets unchanged FP32",
       limitations=["Raw offset magnitude is not causal importance.",
       "Selecting layers by development offset magnitude makes this an exploratory development test.",
       "The current frozen reserved evaluation remains unchanged; results cannot validate a new subset."])
    write(a.output/"partition-audit.json",audit)
    print(json.dumps(audit))
if __name__=="__main__":main()
