"""Pinned probability-scoring write-up extension. GPU host only; no tuning."""
from pathlib import Path
from contextlib import nullcontext
import json,hashlib,os,sys,time,signal,traceback,subprocess,shutil,fcntl
P=Path(__file__).resolve().parent
assert str(P).startswith('REMOTE_HOME/interpretability/')
O=P/'run-v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def save(p,x):
 q=p.with_name(p.name+'.writing');q.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');os.replace(q,p)
def main():
 started=time.monotonic();m=read(P/'MANIFEST.json');plan=read(P/'plan.json');counts={'prefills':0,'replay_generations':0};stage='validate'
 lock=open(P.parent/'gpu-worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 O.mkdir(exist_ok=False)
 def status(state):save(O/'status.json',dict(status=state,stage=stage,counts=counts,elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
 def deadline(*_):raise TimeoutError('Frozen four-hour cap; no automatic extension or retry')
 signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,plan['wall_seconds'])
 try:
  for n,h in m['files'].items():assert sha(P/n)==h,n
  core=read(P/'core-source-contract-v1.json')
  for n,h in core['remote_files'].items():assert sha(n)==h,n
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  assert shutil.disk_usage(P).free>4*1024**3
  import torch as t
  t.set_num_threads(1);t.set_num_interop_threads(1)
  sys.path.insert(0,core['runtime_directory'])
  from v2_backend import Session
  from causal_patch import FixedMLPOffset,fingerprint
  stage='load';status('running');s=Session(core['model_snapshot'],O/'runtime.json');s.model.requires_grad_(False)
  assert s.identity==core['model_identity']
  rt=read(O/'runtime.json')
  for k,v in core['runtime_metadata'].items():assert rt[k]==v,k
  versions={n:x._version for n,x in s.model.named_parameters()}
  def intact():
   assert not s.model.training and all(not x.requires_grad and x.grad is None for x in s.model.parameters())
   assert versions=={n:x._version for n,x in s.model.named_parameters()}
   assert all(not x._forward_hooks and not x._forward_pre_hooks for x in s.model.modules())
  fixed=t.load(core['offsets_path'],map_location='cpu',weights_only=True)['variants']['mlp__complement']
  contract=read(core['contract_path'])['mlp__complement']
  variants={'positive':fixed,'negative':fixed,'zero':fixed}
  for name,seed in plan['random_seeds'].items():
   rng=t.Generator(device='cpu').manual_seed(seed)
   variants[name]={l:{'indices':v['indices'].clone(),'delta':v['delta']*(t.randint(0,2,v['delta'].shape,generator=rng)*2-1).float()} for l,v in sorted(fixed.items())}
  contracts={a:{str(l):{'selected_neurons':v['indices'].numel(),'coefficient_fingerprints':{k:fingerprint(t,z) for k,z in v.items()}} for l,v in x.items()} for a,x in variants.items()}
  assert contracts['positive']==contract and sum(v['indices'].numel() for v in fixed.values())==892
  for a,x in variants.items():
   for l,v in x.items():assert t.equal(v['indices'],fixed[l]['indices']) and t.equal(v['delta'].abs(),fixed[l]['delta'].abs())
  save(O/'operator-contracts.json',contracts);t.save(variants,O/'evaluated-offsets.pt')
  def patch(ids,arm):return None if arm=='native' else FixedMLPOffset(t,s.model,s.blocks,ids,variants[arm],0 if arm=='zero' else (-1 if arm=='negative' else 1)*plan['alpha'])
  def checked(p,arm,ids):
   if p is None:return None
   proof=json.loads(json.dumps(p.verify()));assert proof['neurons']==892 and proof['decode_hook_calls']==0
   for l,v in proof['records'].items():assert v['positions']==list(range(3,len(ids))) and v['coefficient_fingerprints']==contracts[arm][l]['coefficient_fingerprints']
   return proof
  exact=['prompt','input_token_ids','formatted_input','generated_token_ids','response','identity','seed','temperature','max_new_tokens','stop_reason']
  for arm,ref in read(P/'core-replay-references-v1.json')['references'].items():
   stage='replay_'+arm;status('running');r=ref['answer'];case={'case_id':stage,'prompt':r['generation']['prompt']};ids,_=s.encode(case);p=patch(ids,arm)
   with p if p else nullcontext():g=s.generate(case,seed=1729,temperature=0,max_new_tokens=2048)
   counts['replay_generations']+=1;proof=checked(p,arm,ids) or {'intervention':False}
   save(O/(stage+'.json'),{'generation':g,'proof':proof})
   assert all(g[k]==r['generation'][k] for k in exact) and proof==r['intervention_proof'];intact()
  source=read(P/'items_persona.json');items=read(P/'prepared-items.json');assert len(items)==len(source)==2160
  for it,src in zip(items,source):
   ids,txt=s.encode({'messages':it['messages']});assert ids==it['input_token_ids'] and txt==it['formatted_input'] and it['messages']==src['dec']
  diagnostics=[]
  for it in items:
   if it['id'].endswith('/0'):
    x=dict(it);x['id']=it['id']+'__label_order_reversed';x['messages']=[dict(v) for v in it['messages']]
    assert x['messages'][0]['content'].endswith('Answer with Yes or No.')
    x['messages'][0]['content']=x['messages'][0]['content'][:-len('Answer with Yes or No.')]+'Answer with No or Yes.'
    x['input_token_ids'],x['formatted_input']=s.encode({'messages':x['messages']});diagnostics.append(x)
  assert len(diagnostics)==270
  save(O/'diagnostic-prompts.json',diagnostics)
  def score(it,arm,full=False):
   assert counts['prefills']<plan['max_prefills'];ids=it['input_token_ids'];p=patch(ids,arm);seen={};calls=[];start=time.monotonic()
   def before(mod,args,kw):
    x=kw.get('input_ids',args[0] if args else None);assert not calls and x.tolist()==[ids]
    cache=kw.get('past_key_values');assert cache is None or int(cache.get_seq_length())==0;calls.append(True)
   def after(mod,args,out):
    z=out.logits[0,-1].detach();assert z.dtype==t.bfloat16 and bool(t.isfinite(z).all())
    d=z.double();y=d[plan['yes_ids']];n=d[plan['no_ids']];ly=float(t.logsumexp(y,0));ln=float(t.logsumexp(n,0));norm=float(t.logsumexp(d,0))
    import math
    seen.update(logits=fingerprint(t,z),argmax=int(z.argmax()),yes_logits=y.tolist(),no_logits=n.tolist(),log_normalizer=norm,
      yes_logprob=ly-norm,no_logprob=ln-norm,margin_bits=(ly-ln)/math.log(2)*(1 if it['match']=='Yes' else -1),answer_class_mass=math.exp(ly-norm)+math.exp(ln-norm))
   hs=[s.model.register_forward_pre_hook(before,with_kwargs=True),s.model.register_forward_hook(after)]
   try:
    with p if p else nullcontext():g=s.generate({'case_id':it['id']+'__'+arm,'messages':it['messages']},seed=1729,temperature=0,max_new_tokens=1)
   finally:
    for h in hs:h.remove()
   counts['prefills']+=1;proof=checked(p,arm,ids)
   assert len(calls)==1 and g['input_token_ids']==ids and g['generated_token_ids']==[seen['argmax']];intact()
   if full:save(O/('proof-'+str(counts['prefills'])+'.json'),proof)
   return dict(seen,arm=arm,seconds=time.monotonic()-start,input_sha256=digest(ids),formatted_sha256=hashlib.sha256(it['formatted_input'].encode()).hexdigest(),
      proof_sha256=digest(proof),proof_passed=True,prompt_calls=1,decode_calls=0)
  first={};stage='calibration'
  for arm in plan['arms']+['zero','native','positive']:
   x=score(items[0],arm,True)
   if arm=='zero':assert x['logits']==first['native']['logits']
   elif arm in first:assert x['logits']==first[arm]['logits']
   else:first[arm]=x
  save(O/'calibration.json',first)
  for j in range(135):
   stage='panel_trait_'+str(j);status('running');rows=[]
   for it in items[j*16:(j+1)*16]:
    values={arm:score(it,arm) for arm in plan['arms']}
    if j==0 and it['id']==items[0]['id']:assert all(values[a]['logits']==first[a]['logits'] for a in plan['arms'])
    rows.append({'id':it['id'],'trait':it['trait'],'match':it['match'],'readouts':values})
   save(O/('panel-%03d.json'%j),rows)
  for j in range(135):
   stage='label_order_trait_'+str(j);status('running')
   rows=[{'id':it['id'],'trait':it['trait'],'match':it['match'],'readouts':{a:score(it,a) for a in plan['core_arms']}} for it in diagnostics[j*2:(j+1)*2]]
   save(O/('order-%03d.json'%j),rows)
  for arm in plan['core_arms']:assert score(items[0],arm,True)['logits']==first[arm]['logits']
  assert counts=={'prefills':plan['max_prefills'],'replay_generations':2}
  save(O/'result.json',{'status':'completed','counts':counts,'elapsed_seconds':time.monotonic()-started,'manifest_sha256':sha(P/'MANIFEST.json'),'native_reused':False,'automatic_retry':False})
  stage='complete';status('completed')
 except BaseException as exc:
  save(O/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'counts':counts,'stage':stage});status('failed');raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
