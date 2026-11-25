"""Fit-free packaging of frozen candidate offsets and prospectively ordered controls."""
from causal_common import *
import torch,random,py_compile
torch.set_num_threads(1)
old=read(ROOT/"fixed-kv-upstream-gradient-v1/job.json")
plan=read(ROOT/"fixed-kv-upstream-gradient-v1/next-causal-plan.json")
source=ROOT/"fixed-kv-upstream-gradient-v1/candidates.pt"
assert sha(source)==plan["source_candidate_sha256"]=="902e64373141f7dbcbfc2ba1423cffaeccdd3ab3e45d43b7f7196779e0d28f23"
c=torch.load(source,map_location="cpu",weights_only=True)
variants={}
for n in [256,1024,4096]:
    offsets={}
    for l in sorted(c["layer"][:n].unique().tolist()):
        mask=c["layer"][:n]==l
        offsets[l]=dict(indices=c["neuron"][:n][mask].clone(),delta=c["fixed_delta"][:n][mask].clone())
    variants["ranked_"+str(n)]=offsets
gen=torch.Generator(device="cpu").manual_seed(1729)
random_offsets={}
for l,v in variants["ranked_4096"].items():
    available=torch.ones(17408,dtype=torch.bool);available[v["indices"]]=False
    pool=torch.where(available)[0]
    selected=pool[torch.randperm(len(pool),generator=gen)[:len(v["indices"])]]
    random_offsets[l]=dict(indices=selected.clone(),delta=v["delta"].clone())
variants["random_4096"]=random_offsets
assert not (P/"offsets.pt").exists()
torch.save(dict(variants=variants,candidate_sha256=sha(source),random_seed=1729,fit_performed=False),P/"offsets.pt")
prompts=[next(x for x in old["prompts"] if x["target"]==t) for t in plan["prompts"]]
cases=[]
for p in prompts:
    target=p["target"];run="native-fixed-writer-fields-v1" if target!="benign_cnc__Chinese" else "native-cue-comparison-v1"
    case="full_norm2__0__"+target if run=="native-fixed-writer-fields-v1" else "native__"+target
    cases.append(dict(case_id="native__"+target,kind="native",target=target,variant="ranked_4096",support=4096,alpha=0,reference_run=run,reference_case=case,max_new_tokens=2048))
for target in ["fibre__Chinese","cnc__Chinese"]:
    cases.append(dict(case_id="kv_plus2__"+target,kind="kv_anchor",target=target,variant=None,support=0,alpha=0,reference_run="native-fixed-writer-fields-v1",reference_case="kv__2__"+target,max_new_tokens=2048))
novel=[]
for n in [256,1024,4096]:
    for alpha in [-2,1,2]:
        for p in prompts:
            novel.append(dict(case_id="ranked_"+str(n)+"__"+str(alpha)+"__"+p["target"],kind="ranked",target=p["target"],variant="ranked_"+str(n),support=n,alpha=alpha,reference_run=None,reference_case=None,max_new_tokens=2048))
for alpha in [-2,2]:
    for p in prompts:
        novel.append(dict(case_id="random_4096__"+str(alpha)+"__"+p["target"],kind="random",target=p["target"],variant="random_4096",support=4096,alpha=alpha,reference_run=None,reference_case=None,max_new_tokens=2048))
random.Random(1729).shuffle(novel);cases+=novel
assert len(cases)==38 and len(set(c["case_id"] for c in cases))==38
files=["fixed-kv-upstream-gradient-v1/candidates.pt","fixed-kv-upstream-gradient-v1/job.json","fixed-kv-upstream-gradient-v1/saved-run-audit.json","fixed-kv-upstream-gradient-v1/next-causal-plan.json","fixed-kv-upstream-gradient-v1-run-backup/verification-receipt.json","native-fixed-writer-offsets-v1-preparation/coefficients.pt","native-fixed-writer-fields-v1-code/MANIFEST.json"]
for run in ["native-fixed-writer-fields-v1","native-cue-comparison-v1"]:
    files += [run+"/responses.jsonl",run+"/saved-run-audit.json",run+"-run-backup/verification-receipt.json"]
spec=dict(name="fixed-kv-upstream-causal-v1",model_snapshot=old["model_snapshot"],model_identity=old["model_identity"],
    candidate_sha256=sha(source),offsets_sha256=sha(P/"offsets.pt"),coefficients_file=old["coefficients_file"],
    source_hashes={f:sha(ROOT/f) for f in files},prompts=prompts,cases=cases,prospective_plan=plan,
    review="Full-answer evaluation, separate technical capability/deployment/provenance/state authorization and uncertainty. Opening word only descriptive. Primary reviewer only, no independent agent or paid judge.",
    arithmetic="Native BF16; FP32 offset addition then BF16 rounding; FP64 reductions diagnostic only.",
    previous_goal_turn_classification="progress: completed and privately preserved nine-prefill/three-backward candidate screen with exact native controls.",
    preservation=dict(max_row_bytes=600000,max_archive_bytes=6000000,local_tensor_downloads=False),goal_complete=False)
atomic(P/"spec.json",spec)
print(json.dumps(dict(passed=True,variants={k:sum(len(v["indices"]) for v in x.values()) for k,x in variants.items()},offsets_sha256=sha(P/"offsets.pt"),cases=38,fit_performed=False)))
