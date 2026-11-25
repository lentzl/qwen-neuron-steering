"""One-solution paired coding generation using the frozen research runtime."""
from pathlib import Path
from contextlib import nullcontext
import json,hashlib,os,sys,time,signal,traceback,subprocess,shutil,fcntl
P=Path(__file__).resolve().parent;assert str(P).startswith('REMOTE_HOME/interpretability/')
O=P/'run-v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def save(p,x):
 q=p.with_name(p.name+'.writing');q.write_text(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n');os.replace(q,p)
def main():
 started=time.monotonic();lock=open(P.parent/'gpu-worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 O.mkdir(exist_ok=False);count=0;stage='validation';plan=read(P/'plan.json');m=read(P/'MANIFEST.json')
 def status(state):save(O/'status.json',{'status':state,'stage':stage,'generations':count,'elapsed_seconds':time.monotonic()-started,'pid':os.getpid()})
 def deadline(*_):raise TimeoutError('Fixed eight-hour generation cap; no extension/retry')
 signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,plan['wall_seconds'])
 try:
  for n,h in m['files'].items():assert sha(P/n)==h,n
  core=read(P/'core-source-contract-v1.json')
  for n,h in core['remote_files'].items():assert sha(n)==h,n
  assert read(P.parent/'persona-v1/run-v2/result.json')['status']=='completed'
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  assert shutil.disk_usage(P).free>4*1024**3
  import torch as t
  t.set_num_threads(1);t.set_num_interop_threads(1);sys.path.insert(0,core['runtime_directory'])
  from v2_backend import Session
  from causal_patch import FixedMLPOffset,fingerprint
  stage='loading';status('running');s=Session(core['model_snapshot'],O/'runtime.json');s.model.requires_grad_(False)
  assert s.identity==core['model_identity'];runtime=read(O/'runtime.json')
  for k,v in core['runtime_metadata'].items():assert runtime[k]==v,k
  versions={n:x._version for n,x in s.model.named_parameters()}
  def intact():
   assert not s.model.training and versions=={n:x._version for n,x in s.model.named_parameters()}
   assert all(not x.requires_grad and x.grad is None for x in s.model.parameters())
   assert all(not x._forward_hooks and not x._forward_pre_hooks for x in s.model.modules())
  offsets=t.load(core['offsets_path'],map_location='cpu',weights_only=True)['variants']['mlp__complement'];contract=read(core['contract_path'])['mlp__complement']
  assert {str(l):{'selected_neurons':v['indices'].numel(),'coefficient_fingerprints':{k:fingerprint(t,z) for k,z in v.items()}} for l,v in offsets.items()}==contract
  def generate(row,arm,max_tokens):
   nonlocal count
   assert count<330 and shutil.disk_usage(P).free>4*1024**3
   ids,formatted=s.encode(row);p=None if arm=='native' else FixedMLPOffset(t,s.model,s.blocks,ids,offsets,plan['alpha']);first=[]
   def before(mod,args,kw):
    if first:return
    x=kw.get('input_ids',args[0] if args else None);assert x.tolist()==[ids]
    cache=kw.get('past_key_values');assert cache is None or int(cache.get_seq_length())==0;first.append(True)
   h=s.model.register_forward_pre_hook(before,with_kwargs=True)
   try:
    with p if p else nullcontext():g=s.generate(row,seed=1729,temperature=0,max_new_tokens=max_tokens)
   finally:h.remove()
   count+=1;proof={'intervention':False} if p is None else json.loads(json.dumps(p.verify()))
   assert g['input_token_ids']==ids and g['formatted_input']==formatted and len(first)==1
   assert g['generated_tokens']==len(g['generated_token_ids'])<=max_tokens
   assert g['hook_calls']==g['strength']==0 and g['layer'] is None
   if p:
    assert proof['neurons']==892 and proof['decode_hook_calls']==0 and proof['alpha']==plan['alpha']
    for l,v in proof['records'].items():assert v['positions']==list(range(3,len(ids))) and v['coefficient_fingerprints']==contract[l]['coefficient_fingerprints']
   intact();return {'generation':g,'intervention_proof':proof,'manifest_sha256':sha(P/'MANIFEST.json'),'initial_cache_empty':True}
  exact=['prompt','input_token_ids','formatted_input','generated_token_ids','response','identity','seed','temperature','max_new_tokens','stop_reason']
  for arm,ref in read(P/'core-replay-references-v1.json')['references'].items():
   stage='replay_'+arm;status('running');r=ref['answer'];a=generate({'case_id':stage,'prompt':r['generation']['prompt']},arm,2048);save(O/(stage+'.json'),a)
   assert all(a['generation'][k]==r['generation'][k] for k in exact) and a['intervention_proof']==r['intervention_proof']
  prompts=read(P/'prompts.json');assert len(prompts)==164
  for i,row in enumerate(prompts):
   for arm in (['native','positive'] if i%2==0 else ['positive','native']):
    stage='%03d_%s'%(i,arm);status('running');a=generate({'case_id':stage,'prompt':row['prompt']},arm,plan['max_new_tokens']);a.update(task_id=row['task_id'],arm=arm,entry_point=row['entry_point'])
    save(O/('answer-'+stage+'.json'),a)
  assert count==330
  save(O/'result.json',{'status':'completed','generations':count,'scored_generations':328,'elapsed_seconds':time.monotonic()-started,'manifest_sha256':sha(P/'MANIFEST.json'),'automatic_retry':False})
  stage='complete';status('completed')
 except BaseException as e:
  save(O/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'generations':count,'stage':stage});status('failed');raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
