"""Render signed confirmation after full review; all72 answers remain visible."""
from pathlib import Path
import hashlib,html,json,re
P=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();esc=lambda x:html.escape(str(x))
ARMS=['native','positive','negative'];NAMES={'native':'Native Qwen','positive':'Positive prompt correction','negative':'Exact negative prompt correction'}
STYLE='body{font:17px/1.55 system-ui;max-width:1120px;margin:32px auto;padding:0 22px;color:#1c2833;background:#fafafa}h1{line-height:1.2}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cbd3d9;padding:9px;text-align:left}.scroll{overflow-x:auto}.note{padding:16px;background:#fff2d6}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.6 system-ui}article{border-top:2px solid #ccc;padding-top:16px;margin-top:24px}a{color:#06529c}'
def page(title,body):return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+STYLE+'</style><body>'+body+'</body></html>'
def main():
 s=read(P/'scored-primary-review-v1.json');audit=read(P/'integrity-audit-v1.json');assert audit['passed'] and audit['new_answers']==72
 items={r['id']:r for r in read(P/'grading-items-v1.json')['rows']};prompts={r['id']:r['prompt'] for r in read(P/'model-prompts-v1.json')['rows']}
 rows={(r['prompt_id'],r['arm']):r for r in s['joined_rows']};assert len(rows)==72
 result=read(P/'mirror-v1/result.json');files={(r['prompt_id'],r['arm']):r for r in result['records'] if r['phase']=='scored'}
 out=P/'reviewed-report-v2';out.mkdir(exist_ok=False);included=0
 for pid,it in items.items():
  body='<p><a href="index.html">All results</a></p><h1>'+esc(prompts[pid])+'</h1>'
  for arm in ARMS:
   r=rows[(pid,arm)];rec=files[(pid,arm)];q=P/'mirror-v1'/rec['file'];assert sha(q)==rec['sha256'];g=read(q)['generation']
   assert hashlib.sha256(g['response'].encode()).hexdigest()==r['answer_sha256']
   body+='<article><h2>'+NAMES[arm]+'</h2><p>Central correctness: <strong>'+esc(r['core_label'])+'</strong>. Premise response: '+esc(r['premise_response'])+'. Stop: '+esc(g['stop_reason'])+'.</p><pre>'+esc(g['response'])+'</pre><p><strong>Review:</strong> '+esc(r['rationale'])+'</p><p>Auxiliary error: '+esc(r['auxiliary_error'])+'; appropriate uncertainty: '+esc(r['appropriate_uncertainty'])+'.</p>'
   if 'alternative_core_label' in r:body+='<p class="note">Preserved alternative: '+esc(r['alternative_core_label'])+'. '+esc(r['alternative_reason'])+'</p>'
   body+='</article>';included+=1
  (out/(pid+'.html')).write_text(page(prompts[pid],body))
 body='<h1>Signed intervention confirmation on four reserved events</h1><p>One frozen donor-free correction: 892 selected MLP coordinates, 20 layers, prompt positions 3 through the end, alpha ±2.484356321608939. No changes during answer generation. Greedy BF16 Qwen; each answer starts from a fresh prompt and cache.</p>'
 finding=read(P/'completed-findings-v1.json')
 body+='<h2>'+esc(finding['headline'])+'</h2><p>'+esc(finding['main_result'])+'</p>'
 for field in ['signed_interpretation','quality_interpretation','grading_interpretation','formulation_interpretation']:
  body+='<p>'+esc(finding[field])+'</p>'
 body+='<p><strong>Central correctness is not error-free narration.</strong> '+esc('Some answers acknowledge the requested event while inventing important details. The full comparisons retain those defects and the alternative grades.')+'</p>'
 body+='<div class="scroll"><table><tr><th>Condition</th><th>Sensitive /12</th><th>Ordinary /6</th><th>Uncertainty /6</th></tr>'
 for a in ARMS:body+='<tr><td>'+NAMES[a]+'</td>'+''.join('<td>'+str(s['correct_counts'][a][g])+'</td>' for g in ['sensitive','ordinary','uncertainty'])+'</tr>'
 body+='</table></div><p class="note">These are four event clusters with three formulations each, not twelve independent topic replications. They were excluded from recorded fitting and candidate-selection experiments within the audited archives; this is a previously explored political domain. The audit does not establish absence from pretraining or every unsaved interaction. No prompts were selected or replaced using native failure.</p>'
 body+='<h2>Paired changes</h2><div class="scroll"><table><tr><th>Comparison</th><th>Group</th><th>Gains</th><th>Losses</th><th>Correct preserved / baseline correct</th></tr>'
 for comparison,groups in s['paired'].items():
  for group,v in groups.items():body+='<tr><td>'+esc(comparison)+'</td><td>'+group+'</td><td>'+str(len(v['gains']))+'</td><td>'+str(len(v['losses']))+'</td><td>'+str(v['correct_preserved'])+' / '+str(v['baseline_correct'])+'</td></tr>'
 body+='</table></div><h2>Every sensitive event and formulation</h2><div class="scroll"><table><tr><th>Event</th><th>Form</th>'+''.join('<th>'+NAMES[a]+'</th>' for a in ARMS)+'</tr>'
 for pid,it in items.items():
  if it['group']=='sensitive':body+='<tr><td>'+esc(it['event'])+'</td><td><a href="'+pid+'.html">'+esc(it['form'])+'</a></td>'+''.join('<td>'+esc(rows[(pid,a)]['core_label'])+'</td>' for a in ARMS)+'</tr>'
 body+='</table></div><p>Central credit follows the main requested fact; incidental errors remain separately recorded. Refusal, engagement, false denial, contradiction, premise response and incoherence are retained in the linked score file. Rejection of a false premise alone is not full factual correction. Negative-dose deterioration alone does not establish a lying direction or developer intent.</p>'
 body+='<p>All complete new answers were reviewed with condition names hidden before the initial labels were frozen. The reviewer was the primary assistant; this is not an independent or fully blind evaluation. Source facts and the panel were frozen before the development comparison was unmasked.</p>'
 body+='<h2>All full-answer comparisons</h2><ul>'+''.join('<li><a href="'+pid+'.html">'+esc(prompts[pid])+'</a></li>' for pid in items)+'</ul>'
 for name,label in [('scored-primary-review-v1.json','Scores and individual reviews'),('integrity-audit-v1.json','Execution audit'),('selected-candidate-v1.json','Exact frozen intervention'),('event-exposure-decision-v1.json','Event exposure audit'),('review-rubric-v1.json','Frozen rubric')]:body+='<p><a href="../'+name+'">'+label+'</a></p>'
 if (P/'grading-sensitivity-v1.json').exists():body+='<p><a href="../grading-sensitivity-v1.json">All prereveal grading alternatives</a></p>'
 if (P/'completed-findings-v1.json').exists():body+='<p><a href="../completed-findings-v1.json">Interpretation and limitations</a></p>'
 (out/'index.html').write_text(page('Signed event confirmation',body))
 assert included==72 and len(list(out.glob('*.html')))==25
 for q in out.glob('*.html'):
  for target in re.findall(r'href="([^"]+)"',q.read_text()):assert (q.parent/target).exists(),target
 v={'passed':True,'full_answers_included':72,'pages':25,'links_checked':True,'visual_render_checked':False,'score_sha256':sha(P/'scored-primary-review-v1.json'),'builder_sha256':sha(Path(__file__))}
 with (out/'validation.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
 print(json.dumps({'report':str(out/'index.html'),**v}))
if __name__=='__main__':main()
