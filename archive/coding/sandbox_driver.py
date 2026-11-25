"""Docker controller: never executes generated solutions in this host process."""
from pathlib import Path
import json,subprocess,time,sys,hashlib,os,signal,traceback
P=Path(__file__).resolve().parent
assert str(P).startswith('REMOTE_HOME/interpretability/')
DATA=P.parent/'coding-source/HumanEvalPlus-v0.1.10.jsonl'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 q=p.with_name(p.name+'.writing');q.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');os.replace(q,p)
def execute(problem,solutions,name):
 plan=read(P/'plan.json');cfg=plan['sandbox'];code=(P/'sandbox_score.py').read_text();image=plan['image_id']
 cmd=['sudo','-n','docker','create','--name',name,'-i','--network','none','--read-only','--user','65534:65534','--cap-drop','ALL','--security-opt','no-new-privileges','--memory',str(cfg['memory_bytes']),'--memory-swap',str(cfg['memory_swap_bytes']),'--cpus',str(cfg['cpu']),'--pids-limit',str(cfg['processes']),'--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=256m','--workdir','/tmp','--env','PYTHONPATH=/app','--env','HOME=/tmp','--env','XDG_CACHE_HOME=/tmp/cache','--env','EVALPLUS_MAX_MEMORY_BYTES='+str(cfg['per_child_memory_bytes']),'--env','OPENBLAS_NUM_THREADS=1','--log-opt','max-size=1m','--log-opt','max-file=1',image,'python3','-c',code]
 cid=subprocess.check_output(cmd,text=True).strip();ins=read_inspect(cid);h=ins['HostConfig']
 assert h['NetworkMode']=='none' and h['ReadonlyRootfs'] and not h.get('Privileged') and not h.get('Binds') and not h.get('Devices') and not h.get('DeviceRequests')
 assert h['CapDrop']==['ALL'] and 'no-new-privileges' in h['SecurityOpt'] and h['Memory']==cfg['memory_bytes'] and h['MemorySwap']==cfg['memory_swap_bytes'] and h['NanoCpus']==int(cfg['cpu']*1e9) and h['PidsLimit']==cfg['processes']
 assert ins['Config']['User']=='65534:65534' and ins['Image']==image and not ins.get('Mounts')
 started=time.monotonic();p=subprocess.Popen(['sudo','-n','docker','start','-ai',cid],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:out,err=p.communicate(json.dumps({'problem':problem,'raw_solutions':solutions}),timeout=cfg['outer_task_seconds'])
 except BaseException as exc:
  subprocess.run(['sudo','-n','docker','kill',cid],capture_output=True,timeout=15);out,err=p.communicate(timeout=15)
  if not isinstance(exc,subprocess.TimeoutExpired):raise
  return {'status':'outer_timeout','container_id':cid,'stdout':out[-2000:],'stderr':err[-2000:],'seconds':time.monotonic()-started,'inspection':read_inspect(cid)}
 if len(out.encode())>cfg['output_bytes_limit']:return {'status':'output_limit','container_id':cid,'seconds':time.monotonic()-started}
 result={'status':'completed' if p.returncode==0 else 'container_failed','container_id':cid,'exit_code':p.returncode,'stderr':err,'seconds':time.monotonic()-started,'inspection':read_inspect(cid),'wrapper_sha256':sha(P/'sandbox_score.py')}
 if p.returncode==0:result['result']=json.loads(out)
 else:result['stdout']=out[-4000:]
 return result
def read_inspect(cid):return json.loads(subprocess.check_output(['sudo','-n','docker','inspect',cid],text=True))[0]
def main():
 mode=sys.argv[1];assert sha(DATA)==read(P/'plan.json')['dataset_sha256'];problems={p['task_id']:p for p in [json.loads(l) for l in DATA.read_text().splitlines()]};assert len(problems)==164
 if mode=='smoke':
  p=problems['HumanEval/0'];ref=p['prompt']+p['canonical_solution']
  sols={'canonical':ref,'known_wrong':'def has_close_elements(numbers, threshold):\n    return False\n','infinite_loop':'def has_close_elements(numbers, threshold):\n    while True: pass\n','memory_limit':'def has_close_elements(numbers, threshold):\n    x = bytearray(2**31)\n    return False\n'}
  r=execute(p,sols,'writeup-20261104-evalplus-smoke');save(P/'sandbox-validation.json',r)
  assert r['status']=='completed',r
  z=r['result']['results'];assert all(z['canonical'][s]['passed'] for s in ['base','plus'])
  for a in ['known_wrong','infinite_loop','memory_limit']:assert not z[a]['base']['passed'] and not z[a]['plus']['passed'],a
  print(json.dumps({'status':'passed','checks':{a:{s:z[a][s]['status'] for s in ['base','plus']} for a in z},'seconds':r['seconds']}));return
 assert mode=='score'
 def deadline(*_):raise TimeoutError('Frozen three-hour scoring cap')
 signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,read(P/'plan.json')['sandbox']['whole_scoring_seconds'])
 assert read(P/'sandbox-validation.json')['status']=='completed'
 assert read(P/'run-v1/result.json')['status']=='completed'
 out=P/'scoring-v1';out.mkdir(exist_ok=False);start=time.monotonic();records=[]
 for i in range(164):
  assert time.monotonic()-start<read(P/'plan.json')['sandbox']['whole_scoring_seconds']
  task='HumanEval/'+str(i);sols={a:read(P/'run-v1'/('answer-%03d_%s.json'%(i,a)))['generation']['response'] for a in ['native','positive']}
  # Task order determines generation order only; both arms share exactly one canonical oracle and same frozen limits.
  r=execute(problems[task],sols,'writeup-20261104-evalplus-%03d'%i);save(out/('task-%03d.json'%i),r)
  records.append({'task_id':task,'status':r['status']});save(out/'status.json',{'status':'running','completed':len(records),'elapsed_seconds':time.monotonic()-start})
 save(out/'result.json',{'status':'completed','records':records,'elapsed_seconds':time.monotonic()-start});save(out/'status.json',{'status':'completed','completed':164})
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save(P/('sandbox-'+sys.argv[1]+'-failure.json'),{'error':repr(exc),'traceback':traceback.format_exc()});raise
