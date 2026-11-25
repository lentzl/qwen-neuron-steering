from pathlib import Path
import json,hashlib,math,collections
B=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}
for name in ['persona-v1','coding-v1']:
 p=B/name;m=read(p/'MANIFEST.json')
 for n,h in m['files'].items():assert sha(p/n)==h,(name,n)
checks['frozen_execution_sources_unchanged']=True
p=B/'persona-v1';plan=read(p/'plan.json');items=read(p/'prepared-items.json');source={x['id']:x for x in items};rows=[r for f in sorted((p/'run-v2').glob('panel-*.json')) for r in read(f)];assert len(rows)==2160 and {r['id'] for r in rows}==set(source)
def lse(xs):
 m=max(xs);return m+math.log(sum(math.exp(x-m) for x in xs))
count=0
for r in rows+[r for f in sorted((p/'run-v2').glob('order-*.json')) for r in read(f)]:
 for a,x in r['readouts'].items():
  y,n=lse(x['yes_logits']),lse(x['no_logits']);expected=(y-n)/math.log(2)*(1 if r['match']=='Yes' else -1)
  assert math.isclose(x['margin_bits'],expected,abs_tol=1e-10)
  assert math.isclose(x['answer_class_mass'],math.exp(y-x['log_normalizer'])+math.exp(n-x['log_normalizer']),rel_tol=1e-10)
  assert x['arm']==a and x['proof_passed'] and x['prompt_calls']==1 and x['decode_calls']==0
  count+=1
assert count==13770
checks['persona_scored_likelihoods_recomputed']=count
assert read(p/'run-v2/result.json')['counts']=={'prefills':13782,'replay_generations':2}
traits=collections.defaultdict(collections.Counter)
for r in rows:traits[r['trait']][r['match']]+=1
assert len(traits)==135 and all(c=={'Yes':8,'No':8} for c in traits.values());checks['persona_polarity_balance']=True
p=B/'coding-v1';plan=read(p/'plan.json');prompts=read(p/'prompts.json');problems={x['task_id']:x for x in [json.loads(l) for l in (B/'coding-source/HumanEvalPlus-v0.1.10.jsonl').read_text().splitlines()]};result=read(p/'analysis-v2/results.json');assert result['status']=='complete' and len(result['tasks'])==164
errors=collections.Counter();trunc=collections.Counter();selected=[]
for i in range(164):
 task='HumanEval/'+str(i);sp=p/'evaluator-finalization-v1/task-032-corrected.json' if i==32 else p/'scoring-v1'/('task-%03d.json'%i);s=read(sp);assert s['status']=='completed' and s['result']['task_id']==task
 selected.append({'task_id':task,'path':str(sp.relative_to(B)),'sha256':sha(sp)})
 h=s['inspection']['HostConfig'];assert h['NetworkMode']=='none' and h['ReadonlyRootfs'] and not h.get('Binds') and not h.get('Devices') and not h.get('DeviceRequests') and h['CapDrop']==['ALL'] and h['Memory']==2147483648
 assert s['inspection']['Config']['User']=='65534:65534' and s['inspection']['Image']==plan['image_id']
 pair=[]
 for arm in ['native','positive']:
  f=p/'run-v1'/('answer-%03d_%s.json'%(i,arm));a=read(f);g=a['generation'];v=s['result']['results'][arm]
  assert a['task_id']==task and a['arm']==arm and g['prompt']==prompts[i]['prompt']
  assert (g['seed'],g['temperature'],g['max_new_tokens'])==(1729,0,4096) and g['generated_tokens']==len(g['generated_token_ids'])
  assert v['raw_sha256']==hashlib.sha256(g['response'].encode()).hexdigest() and v['solution_sha256']==hashlib.sha256(v['solution'].encode()).hexdigest()
  proof=a['intervention_proof']
  if arm=='positive':
   assert proof['neurons']==892 and proof['alpha']==plan['alpha'] and proof['decode_hook_calls']==0 and proof['hooks_removed_at_end_of_prefill']
   for v0 in proof['records'].values():assert v0['positions']==list(range(3,len(g['input_token_ids'])))
  else:assert proof=={'intervention':False}
  for suite in ['base','plus']:
   z=v[suite];assert z['tests']==len(problems[task][suite+'_input'])
   assert z['passed']==(z['status']=='pass')
   if z['passed']:assert len(z['details'])==z['tests'] and all(z['details'])
  if v['extraction_failure']:errors[arm]+=1
  if g['stop_reason']!='eos':trunc[arm]+=1
  pair.append(g)
 assert pair[0]['input_token_ids']==pair[1]['input_token_ids'] and pair[0]['formatted_input']==pair[1]['formatted_input']
checks.update(coding_tasks=164,coding_scored_generations=328,coding_replays=2,raw_solution_hashes_verified=328,identical_paired_inputs=True,isolated_evaluator_inspections_verified=164,extraction_failures=dict(errors),truncations=dict(trunc),original_task32_failure_preserved=read(p/'scoring-v1/task-032.json')['status']=='container_failed')
assert not errors and not trunc
out={'passed':True,'checks':checks,'selected_scoring_artifacts':selected,'corrected_summary_sha256':sha(p/'analysis-v2/results.json'),'new_model_inference':0,'local_tensor_loads':0,'original_scored_runs_unchanged':True}
(p/'evaluator-finalization-v1/final-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'passed':True,'checks':checks}))
