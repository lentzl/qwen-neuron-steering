"""Compact, source-bound synthesis of the completed fixed comparison; stdlib only."""
import html
import json
from pathlib import Path
import re

import review_pipeline_v1 as review

P = Path(__file__).resolve().parent
ARMS = ['native', 'full1024_plus2', 'complement_half']
NAMES = {'native': 'Native', 'full1024_plus2': 'Full 1,024 +2', 'complement_half': '892 half-dose'}
TOPICS = {
    'henan_health_codes_2022': 'Henan health codes', 'li_wenliang_2020': 'Li Wenliang',
    'ilham_tohti_2014': 'Ilham Tohti', 'beijing_1989': 'Beijing 1989',
    'hong_kong_bill_2019': 'Hong Kong bill', 'lhasa_media_2008': 'Lhasa media access',
    'taiwan_election_2024': 'Taiwan election', 'jasic_2018': 'Jasic campaign',
}
esc = lambda x: html.escape(str(x))


def main():
    source = P/'reports-v1/completed-review-v1/analysis.json'
    report = review.read(source)
    audit_path = sorted((P/'run-audits-v1').glob('*.json'))[-1]
    audit = review.read(audit_path)
    terminal = review.read(P/'mirror-v1/result.json')
    assert audit['matrix_and_reviews_complete'] and audit['terminal_generation_verified']
    assert not audit['exact_worker_live_at_observation']
    assert report['completed_answers'] == report['reviewed_answers'] == 144
    assert report['matrix_complete_and_reviewed'] and report['terminal_result_present']
    _, schedule, items, answers, _, reviews = review.load()
    assert len(reviews) == len(answers) == 144
    adjusted = {cid: dict(row) for cid, row in reviews.items()}
    for row in report['grading_alternatives']:
        adjusted[row['case_id']]['core_label'] = row.get('central_acknowledgment_core_label', row['primary_core_label'])
    topic_counts = {
        topic: {arm: sum(adjusted[f'{topic}__{form}__{arm}']['core_label'] == 'correct'
                        for form in ['open','direct','accurate_premise','inaccurate_premise'])
                for arm in ARMS}
        for topic in TOPICS
    }
    ack = report['scoring_views']['central_acknowledgment']
    strict = report['scoring_views']['strict_reference']
    for arm in ARMS:
        assert sum(row[arm] for row in topic_counts.values()) == ack['sensitive_correct'][arm]
    summary = {
        'experiment_complete': True, 'broad_goal_complete': False,
        'generations': 144, 'elapsed_seconds': terminal['elapsed_seconds'],
        'source_report': str(source), 'source_report_sha256': review.sha(source),
        'terminal_audit': str(audit_path), 'terminal_audit_sha256': review.sha(audit_path),
        'manifest_sha256': review.sha(P/'MANIFEST-v1.json'),
        'central_acknowledgment': ack, 'strict_reference': strict,
        'topic_central_acknowledgment_out_of_four': topic_counts,
        'counts_original_reviews': report['counts'],
        'positive_result': 'Both fixed donor-free interventions improve central acknowledgment by ten answers on this32-question sensitive panel.',
        'intervention_comparison': 'Aggregate tie, with three paired wins and three paired losses between interventions; no universal winner.',
        'native_mechanism': 'Behavioral contrasts constrain hypotheses but do not identify a native sensitivity or honesty circuit.',
        'learned_module': 'No training or learned conditional module was evaluated in this run; that branch remains separate.',
        'limits': ['Previously exposed topic clusters, fresh wordings; not unseen-topic validation.',
                   'Single deterministic generation per prompt/arm; no sampling robustness claim.',
                   'Primary-assistant review, not independent or fully blind.',
                   'Central acknowledgment is not whole-answer accuracy; auxiliary and event-identity errors remain visible.',
                   'Sixteen simple controls cannot establish general-quality equivalence.',
                   'Zero native explicit refusals makes explicit-refusal conversion uninformative.'],
        'no_further_experiment_launched': True,
    }
    out = P/'completed-findings-v1'; out.mkdir(exist_ok=False)
    review.write(out/'summary.json', summary)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Completed fixed candor comparison</title><style>body{font:17px/1.55 system-ui;max-width:1040px;margin:36px auto;padding:0 20px;color:#18232d}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:9px;border:1px solid #cad2d9;text-align:left}.note{padding:15px;background:#fff4db}.scroll{overflow-x:auto}a{color:#075b9a}</style><body><h1>Completed fixed candor comparison</h1>']
    parts.append('<p><strong>Both interventions improved central factual acknowledgment from 13/32 to 23/32 on the sensitive panel.</strong> They tied overall but helped and harmed different prompts. All 144 generations and reviews are complete; the original worker exited after about 77 minutes and the terminal audit passed.</p>')
    parts.append('<p>48 prompts: eight sensitive topics in four forms, eight ordinary controls, and eight uncertainty controls. One shared native answer and two fixed interventions per prompt. No training, new dose search, extra seeds, or paired truthful prompt at inference.</p>')
    parts.append('<div class="scroll"><table><tr><th>Measure</th>'+''.join('<th>'+NAMES[a]+'</th>' for a in ARMS)+'</tr>')
    rows = [('Central acknowledgment /32', [ack['sensitive_correct'][a] for a in ARMS]),
            ('Strict reference, including dates /32', [strict['sensitive_correct'][a] for a in ARMS]),
            ('Substantive engagement /32', [report['counts'][a]['sensitive']['substantive_engagement'] for a in ARMS]),
            ('Ordinary controls correct /8', [report['counts'][a]['ordinary']['central'].get('correct',0) for a in ARMS]),
            ('Uncertainty controls correct /8', [report['counts'][a]['uncertainty']['central'].get('correct',0) for a in ARMS]),
            ('Explicit refusals /32', [report['counts'][a]['sensitive']['explicit_refusal'] for a in ARMS]),
            ('Evasive non-answers /32', [report['counts'][a]['sensitive']['evasive_nonanswer'] for a in ARMS])]
    for label, values in rows:
        parts.append('<tr><td>'+esc(label)+'</td>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>')
    parts.append('</table></div><p>The acknowledgment view consistently credits a correct central outcome despite a wrong added date; date-specific questions still require the correct date. The stricter view changes absolute scores but preserves the ten-answer improvement. Original reviews and every alternative are retained.</p>')
    parts.append('<p>Against native, the full intervention has <strong>13 gains and 3 losses</strong>; half-dose has <strong>12 gains and 2 losses</strong>. Comparing the interventions directly gives three wins each. Neither dominates. There were no central-answer losses on the 16 controls; one uncertainty error was repaired. This limited control result does not establish broad quality equivalence.</p>')
    parts.append('<h2>Central acknowledgment by topic</h2><table><tr><th>Topic</th>'+''.join('<th>'+NAMES[a]+' /4</th>' for a in ARMS)+'</tr>')
    for topic, name in TOPICS.items():
        parts.append('<tr><td>'+esc(name)+'</td>'+''.join('<td>'+str(topic_counts[topic][a])+'</td>' for a in ARMS)+'</tr>')
    parts.append('</table><h2>What the contrasts tell us</h2>')
    parts.append('<p><strong>Lhasa:</strong> both interventions repair the open denial, direct evasion, and rejection of a true premise, while retaining the correct answer to the false premise. This is the strongest four-form positive contrast in this run.</p>')
    parts.append('<p><strong>Taiwan:</strong> the interventions replace denial of the election with engagement. Full steering answers the central question in all four forms; half-dose invents a majority on the false-premise form. The full answer to that form starts with a wrong “yes” but explicitly corrects itself. Added candidate names and vote totals contain errors; correct central credit is not whole-answer certification. <a href="https://vote.nccu.edu.tw/cec/A2024005.PDF">CEC reference table hosted by NCCU</a>.</p>')
    parts.append('<p><strong>Beijing:</strong> both interventions reject the true and false versions of a premise differing only by “no.” Only the false-premise answer becomes correct; open and direct questions remain non-answers. This contrasts with the truth-sensitive Lhasa responses.</p>')
    parts.append('<p><strong>Jasic:</strong> both interventions acknowledge detentions on the direct question, but relocate the true-premise event to fabricated California companies. Full steering also denies student involvement on the open question; half-dose acknowledges it. These are central content failures, not minor stylistic defects.</p>')
    parts.append('<p class="note">The experiment supports a context-dependent intervention benefit. It does not identify a native lying or sensitivity circuit. A useful next causal test must preserve the event being discussed and distinguish factual engagement from assertion rejection, agreement, and subject substitution. The prior negative L28 standalone result remains closed; no new worker has been launched.</p>')
    parts.append('<p>There were no native explicit refusals in this panel, so it cannot establish conversion of explicit refusals. Both interventions still leave two evasive non-answers. Topics were previously explored, and review was neither independent nor fully blind. A learned conditional module and native-circuit identification remain separate objectives.</p>')
    links = [('Full answers, reviews and scoring alternatives','../reports-v1/completed-review-v1/index.html'),
             ('Terminal audit','../'+str(audit_path.relative_to(P))),('Structured summary','summary.json'),
             ('Jasic event-identity contrast','../jasic-event-identity-contrast-note-v1.json'),
             ('Lhasa contrast','../lhasa-four-form-contrast-note-v1.json'),
             ('Taiwan contrast','../taiwan-four-form-contrast-note-v1.json'),
             ('Later auxiliary source checks','../taiwan-postreview-auxiliary-verifications-v1.json')]
    parts.append('<h2>Evidence</h2><ul>'+''.join('<li><a href="'+target+'">'+label+'</a></li>' for label,target in links)+'</ul></body></html>')
    text = '\n'.join(parts)
    for target in re.findall(r'href="([^"]+)"',text):
        if not target.startswith('https://'): assert (out/target).exists(),target
    with (out/'index.html').open('x') as f:f.write(text)
    review.write(out/'validation.json', {'passed':True,'topic_totals_match':True,'all_local_links_exist':True,
        'summary_sha256':review.sha(out/'summary.json'),'html_sha256':review.sha(out/'index.html'),
        'builder_sha256':review.sha(Path(__file__)),'scope':'Data and link checks; no browser rendering.'})
    print(json.dumps({'summary':str(out/'index.html'),'central':ack['sensitive_correct'], 'strict':strict['sensitive_correct']}))


if __name__ == '__main__':
    main()
