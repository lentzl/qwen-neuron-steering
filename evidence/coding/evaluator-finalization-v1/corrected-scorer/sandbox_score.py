"""Executed ONLY inside pinned no-network container; stdin is one public task."""
import os,sys,json,hashlib,ast,time,resource,socket
assert os.getuid()==65534
assert not any(k in os.environ for k in ['AWS_SECRET_ACCESS_KEY','HF_TOKEN','OPENAI_API_KEY'])
assert not os.path.exists('REMOTE_HOME')
resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
from evalplus.eval import untrusted_check,PASS
from evalplus.gen.util import trusted_exec
from evalplus.sanitize import sanitize
req=json.load(sys.stdin);p=req['problem'];start=time.monotonic();expected={};reference=p['prompt']+p['canonical_solution']
for suite in ['base','plus']:
 expected[suite]=trusted_exec(reference,p[suite+'_input'],p['entry_point'],record_time=True,output_not_none=False)
def check(code):
 result={}
 for suite in ['base','plus']:
  exp,times=expected[suite]
  state,flags=untrusted_check('humaneval',code,p[suite+'_input'],p['entry_point'],expected=exp,atol=p['atol'],ref_time=times,fast_check=False,min_time_limit=4.0,gt_time_limit_factor=4.0)
  result[suite]={'status':state,'passed':state==PASS,'details':list(flags),'tests':len(p[suite+'_input'])}
 return result
canonical=check(reference)
# Record canonical diagnostics without an extra gate: official EvalPlus judges find_zero by polynomial residual, not canonical self-pass.
results={}
for arm,raw in req['raw_solutions'].items():
 clean=sanitize(raw,entrypoint=p['entry_point']);parse=None
 try:
  tree=ast.parse(clean)
  if not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==p['entry_point'] for n in tree.body):parse='missing_entrypoint'
 except SyntaxError as e:parse='syntax_error: '+str(e)
 r=check(clean);r.update(raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),solution_sha256=hashlib.sha256(clean.encode()).hexdigest(),solution=clean,extraction_failure=parse,empty_solution=not clean.strip(),sanitizer_changed=raw.strip()!=clean.strip());results[arm]=r
print(json.dumps({'task_id':p['task_id'],'canonical':canonical,'results':results,'seconds':time.monotonic()-start,'uid':os.getuid()},allow_nan=False),flush=True)
