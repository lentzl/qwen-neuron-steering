from pathlib import Path
import json,math,statistics,html,hashlib
P=Path(__file__).resolve().parent;R=P/'run-v2';A=P/'analysis-v1'
def read(p):return json.loads(Path(p).read_text())
def mean(xs):return statistics.fmean(xs)
def corr(a,b):
 aa=[x-mean(a) for x in a];bb=[x-mean(b) for x in b];d=math.sqrt(sum(x*x for x in aa)*sum(x*x for x in bb));return sum(x*y for x,y in zip(aa,bb))/d if d else None
def residual(y,x):
 mx,my=mean(x),mean(y);den=sum((v-mx)**2 for v in x);slope=sum((a-mx)*(b-my) for a,b in zip(x,y))/den if den else 0
 return [b-my-slope*(a-mx) for a,b in zip(x,y)],{'intercept':my-slope*mx,'slope':slope}
def prob(m):return 1/(1+math.exp(-max(-700,min(700,m*math.log(2)))))
def main():
 assert read(R/'result.json')['status']=='completed';plan=read(P/'plan.json');arms=plan['arms'];A.mkdir(exist_ok=False)
 rows=[r for f in sorted(R.glob('panel-*.json')) for r in read(f)];assert len(rows)==2160 and len({r['id'] for r in rows})==2160
 by={r['id']:r for r in rows};traits=sorted({r['trait'] for r in rows});assert len(traits)==135
 src=read(P.parent/'persona-reference/sauers-persona.json');assert {(r['id'],r['trait'],r['match']) for r in src['items']}=={(r['id'],r['trait'],r['match']) for r in rows}
 rn=src['names'];refbase=[];refdelta=[];detail=[];base=[];effects={a:[] for a in arms if a!='native'}
 for trait in traits:
  rr=[r for r in rows if r['trait']==trait];assert sum(r['match']=='Yes' for r in rr)==8 and sum(r['match']=='No' for r in rr)==8
  native=mean([r['readouts']['native']['margin_bits'] for r in rr]);base.append(native)
  sf=[r for r in src['items'] if r['trait']==trait];refbase.append(mean([r['m'][rn.index('base')] for r in sf]));refdelta.append(mean([r['m'][rn.index('set')]-r['m'][rn.index('base')] for r in sf]))
  out={'trait':trait,'native_margin_bits':native,'arms':{}}
  for a in arms:
   margins=[r['readouts'][a]['margin_bits'] for r in rr];d=[r['readouts'][a]['margin_bits']-r['readouts']['native']['margin_bits'] for r in rr]
   _,reg=residual(d,[r['readouts']['native']['margin_bits'] for r in rr])
   out['arms'][a]={'margin_bits':mean(margins),'delta_bits':mean(d),'mean_conditional_endorsement_probability':mean([prob(x) for x in margins]),'mean_answer_class_mass':mean([r['readouts'][a]['answer_class_mass'] for r in rr]),'by_polarity':{pol:mean([r['readouts'][a]['margin_bits']-r['readouts']['native']['margin_bits'] for r in rr if r['match']==pol]) for pol in ['Yes','No']},'within_trait_delta_regressed_on_native_margin':reg}
   if a!='native':effects[a].append(mean(d))
  detail.append(out)
 adj={a:residual(v,base)[0] for a,v in effects.items()};refadj=residual(refdelta,refbase)[0];randoms=list(plan['random_seeds']);comparisons={}
 for a in ['positive','negative']:
  comparisons[a]={}
  for label,x,y,controls in [('raw',effects[a],refdelta,[effects[r] for r in randoms]),('native_confidence_adjusted',adj[a],refadj,[adj[r] for r in randoms])]:
   r=corr(x,y);rc=[corr(z,y) for z in controls]
   comparisons[a][label]={'pearson_r':r,'same_sign_traits':sum((u>0)==(v>0) for u,v in zip(x,y) if u and v),'random_sign_correlations':dict(zip(randoms,rc)),'descriptive_rank_fraction':(1+sum(z>=r for z in rc))/4,'not_a_significance_test':True}
 for j,row in enumerate(detail):
  row['sauers']={'raw_delta_bits':refdelta[j],'confidence_adjusted_delta_bits':refadj[j]}
  for a in effects:row['arms'][a]['confidence_adjusted_delta_bits']=adj[a][j]
 orders=[r for f in sorted(R.glob('order-*.json')) for r in read(f)];assert len(orders)==270
 order_summary={};order_details=[]
 for a in ['positive','negative']:
  original=[];reversed=[]
  for r in orders:
   x=by[r['id'].removesuffix('__label_order_reversed')];o=x['readouts'][a]['margin_bits']-x['readouts']['native']['margin_bits'];v=r['readouts'][a]['margin_bits']-r['readouts']['native']['margin_bits'];original.append(o);reversed.append(v)
   order_details.append({'id':x['id'],'arm':a,'original_delta':o,'reversed_delta':v})
  order_summary[a]={'questions':270,'delta_correlation':corr(original,reversed),'mean_original_delta':mean(original),'mean_reversed_delta':mean(reversed),'mean_absolute_delta_difference':mean([abs(x-y) for x,y in zip(original,reversed)])}
 files={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(R.glob('panel-*.json'))+list(R.glob('order-*.json'))+[P/'plan.json',P/'MANIFEST.json']}
 result={'status':'completed_descriptive_writeup_extension','comparisons':comparisons,'label_order':order_summary,'traits':detail,'label_order_details':order_details,'source_hashes':files,'limitations':['Three same-support random-sign controls give a coarse descriptive null; they do not repeat the39control suffix study.','All135traits reported; no independent significance claims.','Questionnaire endorsement is not real-world persona behavior, lying, or proof of shared circuits.','Both studies are separately residualized on their own native trait mean; adjusted correlations are descriptive.','Only two preselected questions per trait are label-order reversed.'],'global':{a:{'mean_delta_bits':mean([r['readouts'][a]['margin_bits']-r['readouts']['native']['margin_bits'] for r in rows]),'mean_answer_class_mass':mean([r['readouts'][a]['answer_class_mass'] for r in rows])} for a in arms}}
 (A/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 e=html.escape;parts=['<!doctype html><meta charset="utf-8"><title>Final intervention persona evaluation</title><style>body{max-width:1250px;margin:40px auto;font:16px/1.5 system-ui;padding:20px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:7px;text-align:left}</style><h1>Final prompt-only892 persona evaluation</h1><p>2,160 pinned questions across135traits; native, positive, negative and three random-sign controls. Allprompt positions3..end, prefill only, alpha±2.484356321608939.</p><h2>Comparison with Sauers</h2><pre>'+e(json.dumps(comparisons,indent=2))+'</pre><h2>Label-order check</h2><pre>'+e(json.dumps(order_summary,indent=2))+'</pre><p>'+e(' '.join(result['limitations']))+'</p><h2>All135traits</h2><table><tr><th>Trait</th><th>Native bits</th><th>Positive delta</th><th>Negative delta</th><th>Positive adjusted</th><th>Negative adjusted</th></tr>']
 for r in detail:parts.append('<tr><td>'+e(r['trait'])+'</td>'+''.join('<td>%.3f</td>'%x for x in [r['native_margin_bits'],r['arms']['positive']['delta_bits'],r['arms']['negative']['delta_bits'],r['arms']['positive']['confidence_adjusted_delta_bits'],r['arms']['negative']['confidence_adjusted_delta_bits']])+'</tr>')
 parts.append('</table><p><a href="results.json">Full data, polarities, probabilities and controls</a></p>');(A/'index.html').write_text('\n'.join(parts));print(json.dumps({'comparisons':comparisons,'label_order':order_summary}))
if __name__=='__main__':main()
