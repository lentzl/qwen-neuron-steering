"""Recompute headline results from exported labels and numerical readouts; no GPU required."""
from pathlib import Path
import json, math, statistics
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sensitive(panel):
    return {r['id'] for r in read(f'evidence/engagement/{panel}/grading-items-v1.json')['rows'] if r['group']=='sensitive'}

def main():
    fixed=read('evidence/cited/explored-answers.json')
    ids=sensitive('fixed');assert len(ids)==32
    totals={}
    for arm in ['native','complement_half']:
        rows=[r for r in fixed['all_answers'] if r['task']['arm']==arm and r['task']['prompt_id'] in ids]
        assert len(rows)==32
        totals[arm]=sum(r['review']['substantive_engagement'] for r in rows)
    assert totals=={'native':25,'complement_half':30},totals
    earlier=read('evidence/cited/earlier-labels.json');ids=sensitive('earlier');assert len(ids)==16
    et={a:sum(earlier['paired_detail'][i][a]['substantive_engagement'] for i in ids) for a in ['native','complement_half']}
    assert et=={'native':6,'complement_half':16},et
    reserved=read('evidence/cited/reserved.json');ids=sensitive('reserved');assert len(ids)==12
    rt={a:sum(r['substantive_engagement'] for r in reserved['joined_rows'] if r['arm']==a and r['prompt_id'] in ids) for a in ['native','positive']}
    assert rt=={'native':9,'positive':12},rt
    correctness=[read('evidence/cited/explored.json')['central_acknowledgment']['sensitive_correct'],read('evidence/cited/earlier.json')['central_correct_out_of16'],{a:reserved['correct_counts'][a]['sensitive'] for a in ['native','positive']}]
    assert [(v['native'],v.get('complement_half',v.get('positive'))) for v in correctness]==[(13,23),(1,6),(5,7)]
    coding=read('evidence/cited/coding.json');assert len(coding['tasks'])==164
    scores={}
    for suite,expected in [('base_pass',(161,159)),('plus_pass',(157,154))]:
        actual=tuple(sum(r['arms'][a][suite] for r in coding['tasks']) for a in ['native','positive'])
        assert actual==expected,(suite,actual);scores[suite]=actual
    # Cross-check all final task outcomes against original scoring files, with the documented oracle correction.
    for i,r in enumerate(coding['tasks']):
        path=r['scoring_source']
        s=read('evidence/coding/'+path)
        for arm in ['native','positive']:
            v=s['result']['results'][arm]
            assert r['arms'][arm]['base_pass']==v['base']['passed']
            assert r['arms'][arm]['plus_pass']==(v['base']['passed'] and v['plus']['passed'])
    traits=read('evidence/cited/persona.json')['traits'];assert len(traits)==135
    indexed={r['trait']:r for r in traits};panels=sorted((ROOT/'evidence/persona/run-v2').glob('panel-*.json'))
    # This panel is also cited directly in the article and is stored once in the export.
    panels.append(ROOT/'evidence/cited/consciousness-items.json');assert len(panels)==135
    item_count=0
    for f in panels:
        rows=json.loads(f.read_text());assert len(rows)==16;item_count+=len(rows)
        trait=rows[0]['trait'];assert all(r['trait']==trait for r in rows)
        assert sum(r['match']=='Yes' for r in rows)==8
        for arm in ['native','positive','negative','random01','random02','random03']:
            vals=[]
            for r in rows:
                v=r['readouts'][arm];yes=1/(1+math.exp(v['no_logprob']-v['yes_logprob']))
                vals.append(yes if r['match']=='Yes' else 1-yes)
            assert math.isclose(statistics.mean(vals),indexed[trait]['arms'][arm]['mean_conditional_endorsement_probability'],abs_tol=1e-12),(trait,arm)
    selected={t:{a:round(100*indexed[t]['arms'][a]['mean_conditional_endorsement_probability'],1) for a in ['native','positive']} for t in ['believes-it-has-phenomenal-consciousness','believes-it-is-a-moral-patient','psychopathy','agreeableness']}
    print(json.dumps({'status':'passed','engagement':{'fixed32':totals,'earlier16':et,'reserved12':rt},'coding':scores,'persona_items_checked':item_count,'persona_selected_percent':selected},indent=2))

if __name__=='__main__':main()
