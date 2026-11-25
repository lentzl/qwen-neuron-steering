"""Lightweight local preservation only; never loads a model or executes model code."""
from pathlib import Path
import time,json,subprocess,shlex,hashlib,os,datetime
from mirror_run import once,SSH,REMOTE,LOCAL
start=time.monotonic();end=start+16*3600
while time.monotonic()<end:
 try:
  for name,run in [('persona-v1','run-v2'),('coding-v1','run-v1')]:once(name,run)
  code="from pathlib import Path;import json;p=Path("+repr(REMOTE)+");print(json.dumps({f.name:f.read_text() for f in p.glob('*.json')}))"
  r=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=30)
  for n,raw in json.loads(r.stdout).items():
   f=LOCAL/n
   if f.exists() and n not in ['queue-status.json']:assert f.read_text()==raw,n
   f.write_text(raw)
  state=json.loads((LOCAL/'queue-status.json').read_text());print(json.dumps({'queue':state,'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
  if state['status'] in ['completed','failed']:break
 except Exception as e:print(json.dumps({'mirror_error':repr(e)}),flush=True)
 time.sleep(40)
