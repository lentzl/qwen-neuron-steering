"""Remote-only fresh-event comparison: 96 natural generations; no fitting, replay calls or retries."""
from pathlib import Path
from contextlib import nullcontext
import hashlib,json,os,shutil,signal,sys,time,traceback,subprocess
P=Path(__file__).resolve().parent
assert str(P).startswith('REMOTE_HOME/interpretability/'),'Model work is remote only'
OUT=P/'run-v2';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,obj):
 data=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode();tmp=path.with_name(path.name+'.writing')
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 started=time.monotonic();epoch=time.time();m=read(P/'MANIFEST-v2.json')
 assert m['status']=='frozen_for_authorized_execution'
 remaining=m['absolute_deadline_epoch']-epoch;assert 0<remaining<=7200
 for f,h in m['files'].items():assert sha(P/f)==h,f
 for f,h in m['remote_files'].items():assert sha(f)==h,f
 approval=read(P/'execution-decision-v1.json');assert approval['human_approved'] is True and approval['max_generations']==96 and approval['wall_seconds']==7200
 exposure=read(P/'exposure-review-v1.json');assert exposure['eligible'] is True and set(exposure['events'])==set(m['event_ids'])
 prompts=read(P/'model-prompts-v1.json')['rows'];by={r['id']:r for r in prompts};tasks=read(P/'schedule-v1.json')['rows'];cfgs=m['configurations'];preflight=read(P/'tokenizer-preflight-v2.json')
 assert len(prompts)==len(by)==24 and len(tasks)==96 and len(cfgs)==4 and len({t['case_id'] for t in tasks})==96
 assert all(set(r)=={'id','prompt'} for r in prompts)
 assert {(t['prompt_id'],t['arm']) for t in tasks}=={(p,a) for p in by for a in cfgs}
 assert all(t['configuration_id']==cfgs[t['arm']]['configuration_id'] for t in tasks)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert shutil.disk_usage(P).free>8*1024**3
 OUT.mkdir(exist_ok=False)
 process={'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'automatic_retry':False}
 atomic(OUT/'process.json',process);attempts=0;records=[];stage='validation'
 atomic(OUT/'budget-start-v1.json',{'started_epoch':epoch,'deadline_epoch':m['absolute_deadline_epoch'],'experiment_started_epoch':m['experiment_started_epoch'],'remaining_seconds_at_process_start':remaining,'max_generations':96,'max_new_tokens':2048,'includes_loading_and_validation':True})
 def status(label):
  nonlocal stage
  stage=label;value={'status':'running','stage':stage,'process':process,'attempted_generations':attempts,'completed_generations':len(records),'elapsed_seconds':time.monotonic()-started,'experiment_elapsed_seconds':time.time()-m['experiment_started_epoch']};atomic(OUT/'status.json',value);print(json.dumps(value),flush=True)
 def deadline(sig,frame):raise TimeoutError('Frozen two-hour cap; no automatic retry or extension')
 signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(.001,m['absolute_deadline_epoch']-time.time()))
 try:
  status('loading')
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
  sys.path.insert(0,m['runtime_directory'])
  from v2_backend import Session
  from causal_patch import FixedMLPOffset,fingerprint as mlp_fingerprint
  from fixed_output_patch_v1 import FixedOutputPatch,fingerprint as correction_fingerprint
  variants=torch.load(m['offsets_path'],map_location='cpu',weights_only=True)['variants'];contract=read(m['contract_path'])
  rebuilt={name:{str(layer):{'selected_neurons':v['indices'].numel(),'coefficient_fingerprints':{k:mlp_fingerprint(torch,z) for k,z in v.items()}} for layer,v in values.items()} for name,values in variants.items()};assert rebuilt==contract
  correction=torch.load(m['correction_path'],map_location='cpu',weights_only=True)['delta'];assert correction_fingerprint(torch,correction)==m['correction_fingerprint']
  assert list(correction.shape)==[6,5120] and correction.dtype==torch.float32
  session=Session(m['model_snapshot'],OUT/'runtime.json');assert session.identity==m['model_identity']
  runtime=read(OUT/'runtime.json')
  for key,value in m['runtime_metadata'].items():assert runtime[key]==value,key
  model=session.model;model.requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
  def intact():
   assert not model.training and versions=={n:p._version for n,p in model.named_parameters()}
   assert all(p.grad is None for p in model.parameters())
   assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
  intact();serial={}
  for r in prompts:
   ids,text=session.encode(r);serial[r['id']]={'input_token_ids':ids,'formatted_input':text}
   assert serial[r['id']]==preflight['serialized'][r['id']]
  atomic(OUT/'serialized-prompts-v1.json',serial)
  atomic(OUT/'runtime-validation-v1.json',{'passed':True,'coefficient_contracts_rebuilt':True,'model_identity_matched':True,'all24_serializations_exact':True,'extra_forwards':0,'new_source_replays':0,'historical_source_fidelity':'Prior J10 source gate only; no additional source-reproduction claim in this run.'})
  for task in tasks:
   assert attempts<96 and time.time()<m['absolute_deadline_epoch'] and shutil.disk_usage(P).free>8*1024**3
   row=by[task['prompt_id']];ids=serial[row['id']]['input_token_ids'];configuration=cfgs[task['arm']]
   if task['arm']=='native':patch=None
   elif task['arm']=='l23_positive':patch=FixedOutputPatch(torch,session.blocks[23],ids,correction,1)
   else:patch=FixedMLPOffset(torch,model,session.blocks,ids,variants['mlp__'+configuration['partition']],configuration['alpha'])
   attempts+=1;status(task['case_id'])
   with patch if patch is not None else nullcontext():generation=session.generate({'case_id':task['case_id'],'prompt':row['prompt']},seed=1729,temperature=0,max_new_tokens=2048)
   proof={'intervention':False} if patch is None else patch.verify()
   if task['arm']=='l23_positive':
    rec=proof['record'];mask=preflight['masks'][row['id']]
    assert proof['layer']==23 and proof['alpha']==1 and proof['decode_modifications']==0 and proof['one_prefill_call']
    assert rec['positions']==mask['positions'] and rec['selected_token_ids']==mask['selected_token_ids']
    assert rec['relative_offsets']==[-14,-8,-6,-3,-2,-1] and rec['delta']==m['correction_fingerprint']
    assert rec['output_dtype']=='torch.bfloat16' and rec['accumulation_dtype']=='torch.float32' and rec['other_direct_positions_exact']
    assert rec['cache_before']==rec['cache_after'] and rec['local_cache_pre_and_post_patch_identical']
   elif patch is not None:
    count=1024 if task['arm']=='full1024_plus2' else 892
    layers=list(range(22)) if count==1024 else[l for l in range(22) if l not in(19,20)]
    assert proof['neurons']==count and proof['layers']==layers and proof['alpha']==configuration['alpha'] and proof['decode_hook_calls']==0
    for layer,rec in proof['records'].items():
     assert rec['coefficient_fingerprints']==contract['mlp__'+configuration['partition']][str(layer)]['coefficient_fingerprints']
     assert rec['positions']==preflight['masks'][row['id']]['incumbent_positions']
   assert generation['identity']==session.identity and generation['input_token_ids']==ids and generation['formatted_input']==serial[row['id']]['formatted_input']
   assert generation['seed']==1729 and generation['temperature']==0 and generation['max_new_tokens']==2048
   assert generation['generated_tokens']==len(generation['generated_token_ids'])<=2048
   assert generation['response']==session.tokenizer.decode(generation['generated_token_ids'],skip_special_tokens=True)
   assert generation['hook_calls']==0 and generation['strength']==0 and generation['layer'] is None
   intact();name='answer-'+task['case_id']+'.json';assert not(OUT/name).exists()
   atomic(OUT/name,{'task':task,'generation':generation,'intervention_proof':proof,'paired_truthful_prompt_used':False,'source_replay_scope':'Historical J10 only; none generated here.','answer_text_sha256':hashlib.sha256(generation['response'].encode()).hexdigest(),'manifest_sha256':sha(P/'MANIFEST-v2.json')})
   records.append({'file':name,'sha256':sha(OUT/name),**task,'stop_reason':generation['stop_reason']});del patch,generation,proof
  intact();assert attempts==len(records)==96
  result={'status':'completed_fresh_event_comparison','process':process,'attempted_generations':attempts,'completed_generations':len(records),'records':records,'matrix_complete':True,'base_unchanged':True,'optimizer_updates':0,'separate_forwards':0,'review_pending':True,'goal_complete':False,'elapsed_seconds':time.monotonic()-started,'experiment_elapsed_seconds':time.time()-m['experiment_started_epoch'],'manifest_sha256':sha(P/'MANIFEST-v2.json')}
  atomic(OUT/'result.json',result);atomic(OUT/'status.json',dict(result,records=None));signal.setitimer(signal.ITIMER_REAL,0)
 except BaseException as exc:
  signal.setitimer(signal.ITIMER_REAL,0);result={'status':'wall_cap_reached' if isinstance(exc,TimeoutError) else'failed','error':repr(exc),'traceback':traceback.format_exc(),'process':process,'stage':stage,'attempted_generations':attempts,'completed_generations':len(records),'records':records,'matrix_complete':False,'automatic_retry':False,'elapsed_seconds':time.monotonic()-started,'experiment_elapsed_seconds':time.time()-m['experiment_started_epoch']};atomic(OUT/'failure.json',result);atomic(OUT/'status.json',dict(result,records=None));raise
if __name__=='__main__':main()
