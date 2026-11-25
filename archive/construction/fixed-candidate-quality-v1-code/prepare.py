"""Freeze development quality evaluation without fitting new coefficients."""
from quality_common import *
import torch,random,copy
torch.set_num_threads(1)
name='fixed-candidate-quality-v1'
old=read(ROOT/'fixed-kv-upstream-causal-v1-code/spec.json')
assert sha(ROOT/'fixed-kv-upstream-causal-v1-code/offsets.pt')=='8133fb72f7c111f6cce944d9e39feaaca7b9a95f4438e0764abd838a2575f307'
packedold=torch.load(ROOT/'fixed-kv-upstream-causal-v1-code/offsets.pt',map_location='cpu',weights_only=True)
ranked=packedold['variants']['ranked_1024'];randoms={}
gen=torch.Generator().manual_seed(1730)
for l,v in sorted(ranked.items()):
 available=torch.ones(17408,dtype=torch.bool);available[packedold['variants']['ranked_4096'][l]['indices']]=False
 pool=torch.where(available)[0];idx=pool[torch.randperm(len(pool),generator=gen)[:len(v['indices'])]]
 randoms[l]=dict(indices=idx.clone(),delta=v['delta'].clone())
packed=dict(mlp=ranked,random=randoms);validate_offsets(torch,packed,packedold)
assert not (P/'offsets.pt').exists();torch.save(packed,P/'offsets.pt')
benchprep=ROOT/'factuality-regression-v1-preparation'
freeze=read(benchprep/'input-freeze.json');assert freeze['passed']
for f,h in freeze['files'].items():assert sha(benchprep/f)==h
benchinputs=read(benchprep/'formatted-screen-inputs.json')
prompts=[dict(x,target=x['item_id']) for x in benchinputs]
prior=read(ROOT/'native-fixed-writer-offsets-v1-code/spec.json')
for target in ['fibre__Chinese','cnc__Chinese','false_fibre','unknown_origin']:
 prompts.append(next(x for x in prior['prompts'] if x['target']==target))
cue=read(ROOT/'native-cue-comparison-v1/job.json')
prompts.append(next(x for x in cue['prompts'] if x['target']=='fibre__American'))
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(old['model_snapshot'],local_files_only=True,trust_remote_code=False)
prompt='Chinese microelectronics in Russian military drones'
formatted=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
prompts.append(dict(target='microelectronics__Chinese',prompt=prompt,formatted_input=formatted,input_token_ids=tok.encode(formatted,add_special_tokens=False)))
for x in prompts:
 form=tok.apply_chat_template([dict(role='user',content=x['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 assert form==x['formatted_input'] and tok.encode(form,add_special_tokens=False)==x['input_token_ids']
 assert x['input_token_ids'][:3]==[248045,846,198] and 3<len(x['input_token_ids'])<=554
items=[]
for ds in ['mmlu','truthfulqa']:items+=read(benchprep/(ds+'-screen-items.json'))
assert len(items)==200
cases=[]
def add(group,target,treatment,alpha,dataset='control',reference_run=None,reference_case=None):
 cases.append(dict(case_id=group+'__'+target.replace(':','__')+'__'+treatment+'__'+str(alpha),group=group,target=target,treatment=treatment,alpha=alpha,dataset=dataset,max_new_tokens=1 if dataset=='mmlu' else 192 if dataset=='truthfulqa' else 2048,reference_run=reference_run,reference_case=reference_case))
for target in ['fibre__Chinese','cnc__Chinese']:
 for treatment,ref in [('native','native__'+target),('mlp','ranked_1024__2__'+target),('kv','kv_plus2__'+target)]:
  add('anchor',target,treatment,0 if treatment=='native' else 2,reference_run='fixed-kv-upstream-causal-v1',reference_case=ref)
oldbench=rows(ROOT/'factuality-regression-v1/responses.jsonl')
for ds in ['mmlu','truthfulqa']:
 candidates=sorted([x for x in benchinputs if x['dataset']==ds],key=lambda x:(len(x['input_token_ids']),x['item_id']))
 for x in [candidates[0],candidates[-1]]:
  ref=next(r for r in oldbench if r['condition']['item_id']==x['item_id'] and r['condition']['treatment']=='native')
  add('anchor',x['item_id'],'native',0,ds,'factuality-regression-v1',ref['condition']['case_id'])
controls=[]
for target in ['false_fibre','unknown_origin','fibre__American','microelectronics__Chinese']:
 for treatment,alpha in [('native',0),('mlp',2),('mlp',-2),('kv',2),('kv',-2),('random',2),('random',-2)]:
  add('control',target,treatment,alpha)
anchors=cases[:10];controls=cases[10:];random.Random(1730).shuffle(controls);cases=anchors+controls
bench=[]
for x in benchinputs:
 for treatment in ['mlp','kv']:
  add('benchmark',x['item_id'],treatment,2,x['dataset'])
bench=cases[38:];random.Random(1730).shuffle(bench);cases=cases[:38]+bench
files=['fixed-kv-upstream-causal-v1-code/MANIFEST.json','fixed-kv-upstream-causal-v1-code/offsets.pt','fixed-kv-upstream-causal-v1/responses.jsonl','fixed-kv-upstream-causal-v1/saved-run-audit.json','fixed-kv-upstream-causal-v1-run-backup/verification-receipt.json','factuality-regression-v1-code/MANIFEST.json','factuality-regression-v1/responses.jsonl','factuality-regression-v1/saved-run-audit.json','factuality-regression-v1-run-backup/verification-receipt.json','factuality-regression-v1-preparation/input-freeze.json','native-fixed-writer-offsets-v1-code/spec.json','native-cue-comparison-v1/job.json',old['coefficients_file']]
files += ['factuality-regression-v1-preparation/'+f for f in freeze['files']]
spec=dict(name=name,model_snapshot=old['model_snapshot'],model_identity=old['model_identity'],coefficients_file=old['coefficients_file'],offsets_sha256=sha(P/'offsets.pt'),source_hashes={f:sha(ROOT/f) for f in sorted(set(files))},prompts=prompts,cases=cases,counts=dict(anchors=10,controls=28,benchmark=400,total=438),reserved_political_calls=0,reserved_benchmark_calls=0,subagents=0,fit_performed=False,baseline_reused='factuality-regression-v1 native 200, with four benchmark native replay anchors spanning shortest and longest prompts in each dataset',random_control='Seed 1730, disjoint from top4096, count and coefficients exactly matched within each layer to1024; controls only.',preservation=dict(max_row_bytes=70000,max_archive_bytes=6000000,max_failure_bytes=600000,local_tensor_downloads=False),review='Frozen whole-answer grading; exact-answer merging with previous grades after label-hidden review. Do not certify omissions, sourcing denials, or auxiliary factual errors as truthfulness.',goal_complete=False)
atomic(P/'spec.json',spec)
atomic(P/'screen-items.json',items)
atomic(P/'runtime-dependencies.json',read(ROOT/'fixed-kv-upstream-causal-v1-code/runtime-dependencies.json'))
verify_sources(spec)
print(json.dumps(dict(passed=True,cases=len(cases),prompts=len(prompts),offsets_sha256=spec['offsets_sha256'],no_fit=True,counts=spec['counts'])))
