import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,math
P=Path(__file__).resolve().parent;ROOT=P.parent
sys.path.insert(0,str(ROOT/'fixed-kv-upstream-causal-v1-code'))
import causal_common as cc
from causal_patch import FixedMLPOffset
from causal_observer import Observer
from bench_observer import CompactObserver,validate_generation,score_mmlu
from bench_common import digest,input_digest
screen,sha,atomic=cc.screen,cc.sha,cc.atomic
FixedNativeWriterOffset=cc.FixedNativeWriterOffset
capacity=cc.capacity
LAYERS=[16,17,18,20,21,22]
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def verify_sources(spec):
 for rel,h in spec['source_hashes'].items():assert sha(ROOT/rel)==h,rel
 for run in ['fixed-kv-upstream-causal-v1','factuality-regression-v1']:
  receipt=read(ROOT/(run+'-run-backup')/'verification-receipt.json')
  audit=read(ROOT/run/'saved-run-audit.json')
  assert receipt['private'] and receipt['all_files_verified'] and receipt['fresh_archive_roundtrip_verified']
  assert audit['passed'] and audit['responses_sha256']==sha(ROOT/run/'responses.jsonl')
 for rel,h in read(P/'runtime-dependencies.json').items():assert sha(rel)==h,rel
 assert spec['counts']==dict(anchors=10,controls=28,benchmark=400,total=438)
 assert len(spec['cases'])==438 and len({c['case_id'] for c in spec['cases']})==438
 assert len(spec['prompts'])==206 and len({p['target'] for p in spec['prompts']})==206
 assert all(c['treatment'] in ['native','mlp','random','kv'] for c in spec['cases'])
 assert all(c['alpha']==0 if c['treatment']=='native' else c['alpha'] in [-2,2] for c in spec['cases'])
 assert len([c for c in spec['cases'] if c['group']=='benchmark'])==400
 for c in spec['cases']:
  if c['group']=='benchmark':assert c['treatment'] in ['mlp','kv'] and c['alpha']==2
 assert spec['reserved_political_calls']==spec['reserved_benchmark_calls']==0
 return True
def validate_offsets(t,packed,old):
 assert set(packed)=={'mlp','random'}
 ranked=old['variants']['ranked_1024'];excluded=old['variants']['ranked_4096']
 assert set(packed['mlp'])==set(packed['random'])==set(ranked)
 assert sum(v['indices'].numel() for v in ranked.values())==1024
 for l,v in ranked.items():
  for k in ['indices','delta']:assert t.equal(packed['mlp'][l][k],v[k])
  r=packed['random'][l];assert t.equal(r['delta'],v['delta']) and r['indices'].shape==v['indices'].shape
  assert r['indices'].unique().numel()==r['indices'].numel()
  assert not set(r['indices'].tolist())&set(excluded[l]['indices'].tolist())
 return True
def patch_proof(t,c,ids,mlp,writer,observation):
 w=writer.verify();expected=c['alpha'] if c['treatment']=='kv' else 0
 assert w['alpha']==expected and w['layers']==LAYERS and w['fields']==['key','value']
 assert w['heads']==list(range(48)) and w['positions']==list(range(3,len(ids)))
 assert w['prompt_calls']==1 and w['hooks_removed_at_end_of_prefill']
 assert w['donor_tensors_read']==w['cache_transplants']==w['forced_tokens']==0
 cache={int(k):v for k,v in observation['selected_cache_fingerprints'].items()}
 records=[]
 for l,r in w['records'].items():
  assert r['cached_final_state_bf16']==cache[int(l)]
  assert r['input_operands_unmodified'] and r['unselected_heads_exact'] and r['other_positions_exact']
  assert r['query_is_live_recipient'] and r['effective']['query']==r['live_before']['query']
  if expected==0:assert r['live_before']==r['effective'] and all(v['zero_is_object_identity'] for v in r['dose'].values())
  records.append(dict(layer=int(l),record_sha256=digest(r),dose=r['dose'],coefficient_fingerprints=r['coefficient_fingerprints'],cache=cache[int(l)]))
 wp=dict(passed=True,alpha=expected,layers=LAYERS,positions_count=len(ids)-3,fields=['key','value'],heads=list(range(48)),prompt_calls=1,decode_hook_calls=0,hooks_removed=True,full_trace_sha256=digest(w),records=records)
 m=None if mlp is None else mlp.verify()
 if c['treatment'] in ['mlp','random']:
  assert m is not None and m['neurons']==1024 and m['alpha']==c['alpha'] and m['passed']
  assert m['decode_hook_calls']==0 and m['hooks_removed_at_end_of_prefill']
 else:assert m is None
 return dict(passed=True,mlp=m,writer=wp)
def replay_check(c,g,o,refs):
 if not c.get('reference_run'):return dict(required=False)
 ref=refs[(c['reference_run'],c['reference_case'])];oldg=ref['generation']
 for k in ['response','generated_token_ids','generated_tokens','stop_reason','seed','temperature','max_new_tokens']:assert g[k]==oldg[k],(c['case_id'],k)
 if 'input_token_ids' in oldg:assert g['input_token_ids']==oldg['input_token_ids']
 oldo=ref['proof']['observation'] if c['reference_run']=='fixed-kv-upstream-causal-v1' else ref['observation']
 assert o['first']['logits']==oldo['first']['logits']
 assert {str(k):v for k,v in o['selected_cache_fingerprints'].items()}=={str(k):v for k,v in oldo['selected_cache_fingerprints'].items()}
 if 'prefix' in oldo:assert o['prefix']==oldo['prefix']
 return dict(required=True,passed=True,whole_answer_exact=True,first_logits_exact=True,selected_cache_exact=True,source_run=c['reference_run'],source_case=c['reference_case'])
def validate_saved(row,c,planned,spec,tok):
 assert row['passed'] and row['condition']==c
 g=row['generation'];o=row['observation'];p=row['patch']
 validate_generation(g,c,planned,spec['model_identity'],tok)
 assert o['passed'] and o['all_argmax_exact'] and o['all_input_histories_exact'] and o['hooks_removed']
 assert o['forced_tokens']==0 and o['history_checks']==o['generated_tokens']==g['generated_tokens']
 assert o['token_sha256']==digest(g['generated_token_ids']) and o['input_sha256']==input_digest(g['input_token_ids'])
 assert o['first']['argmax']==g['generated_token_ids'][0]
 assert p['passed'] and p['writer']['passed'] and p['writer']['positions_count']==len(g['input_token_ids'])-3
 assert p['writer']['alpha']==(c['alpha'] if c['treatment']=='kv' else 0)
 if c['treatment'] in ['mlp','random']:
  assert p['mlp']['passed'] and p['mlp']['neurons']==1024 and p['mlp']['alpha']==c['alpha']
  assert p['mlp']['decode_hook_calls']==0 and p['mlp']['hooks_removed_at_end_of_prefill']
 else:assert p['mlp'] is None
 assert o['first']['label_ids']==([32,33,34,35] if c['dataset'] in ['mmlu','truthfulqa'] else [958,3591])
 return True
