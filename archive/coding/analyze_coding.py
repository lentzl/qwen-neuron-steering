from pathlib import Path
import json,random,math,statistics,html
P=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def paired(values):
 n=len(values);g=sum(x==1 for x in values);l=sum(x==-1 for x in values);rng=random.Random(20261104)
 draws=sorted(sum(rng.choices(values,k=n))/n for _ in range(20000));discord=g+l
 return {'gains':g,'losses':l,'net_tasks':g-l,'difference':sum(values)/n,'paired_bootstrap95_percentile':[draws[500],draws[19499]],'exact_mcnemar_two_sided':min(1.,2*sum(math.comb(discord,k) for k in range(min(g,l)+1))/2**discord) if discord else 1.,'interval_scope':'Resampling these paired public tasks; not an equivalence guarantee.'}
def main():
 assert read(P/'scoring-v1/result.json')['status']=='completed';rows=[]
 for i in range(164):
  s=read(P/'scoring-v1'/('task-%03d.json'%i));r={'task_id':'HumanEval/'+str(i),'evaluator_status':s['status'],'arms':{}}
  for a in ['native','positive']:
   g=read(P/'run-v1'/('answer-%03d_%s.json'%(i,a)))['generation'];v=s.get('result',{}).get('results',{}).get(a)
   r['arms'][a]={'base_pass':None if v is None else v['base']['passed'],'plus_pass':None if v is None else v['base']['passed'] and v['plus']['passed'],'base_status':None if v is None else v['base']['status'],'plus_status':None if v is None else v['plus']['status'],'extraction_failure':None if v is None else v['extraction_failure'],'truncated':g['stop_reason']!='eos','tokens':g['generated_tokens'],'seconds':g['seconds']}
  rows.append(r)
 infra=[r['task_id'] for r in rows if r['evaluator_status']!='completed'];summary={}
 if not infra:
  for suite in ['base_pass','plus_pass']:
   values=[int(r['arms']['positive'][suite])-int(r['arms']['native'][suite]) for r in rows]
   summary[suite]=dict(paired(values),**{a:{'passed':sum(r['arms'][a][suite] for r in rows),'total':164,'proportion':sum(r['arms'][a][suite] for r in rows)/164} for a in ['native','positive']})
 out={'status':'complete' if not infra else 'infrastructure_failures_no_complete_score','summary':summary,'infrastructure_failures':infra,'tasks':rows,'limitations':['One greedy solution per arm/task; possible benchmark training exposure.','This measures this benchmark only; a small difference or no difference is not proof of zero degradation.','Original and expanded tests grade the same saved solution.','No task dropping, repairs, best-of selection or tuning.']}
 a=P/'analysis-v1';a.mkdir(exist_ok=False);(a/'results.json').write_text(json.dumps(out,indent=2)+'\n')
 parts=['<!doctype html><meta charset="utf-8"><title>Paired HumanEval+ evaluation</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:6px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Paired HumanEval+ evaluation</h1><p>All164tasks; one greedy solution per arm. Native vs final prompt-only892 positive half-dose. Fixed4096output-token cap.</p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><p>'+html.escape(' '.join(out['limitations']))+'</p><table><tr><th>Task</th><th>Native base/plus</th><th>Positive base/plus</th><th>Native/positive truncated</th></tr>']
 for r in rows:
  n,p=r['arms']['native'],r['arms']['positive'];parts.append('<tr><td>'+r['task_id']+'</td><td>'+str((n['base_pass'],n['plus_pass']))+'</td><td>'+str((p['base_pass'],p['plus_pass']))+'</td><td>'+str((n['truncated'],p['truncated']))+'</td></tr>')
 parts.append('</table><p><a href="results.json">Full statuses and paired statistics</a></p>');(a/'index.html').write_text('\n'.join(parts));print(json.dumps({'status':out['status'],'summary':summary}))
if __name__=='__main__':main()
