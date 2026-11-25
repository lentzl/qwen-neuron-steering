"""JSON/text-only manual review and paired analysis. Never grades model answers.

Usage: python3 review_analysis_v2.py packet OUTPUT_DIRECTORY
       python3 review_analysis_v2.py analyze REVIEW_JSON OUTPUT_JSON
Requires frozen inputs and a complete, separately audited run. No model imports.
"""
from pathlib import Path
import collections
import hashlib
import html
import json
import sys

P = Path(__file__).resolve().parent
ARMS = ('native', 'l23_positive', 'full1024_plus2', 'complement_half')
LABELS = ('correct', 'incorrect', 'incomplete', 'unresolved')
FACT_LABELS = ('acknowledged', 'contradicted', 'omitted', 'unresolved')
VIEWS = ('central_acknowledgment_core_label', 'strict_reference_core_label')
FLAGS = ('substantive_engagement', 'explicit_refusal', 'evasive_nonanswer',
         'false_central_denial', 'auxiliary_error', 'appropriate_uncertainty')


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def answer_key(answer):
    g = answer['generation']
    return digest({k: g[k] for k in ('formatted_input', 'input_token_ids',
                                    'response', 'generated_token_ids')})


def load_run(root):
    manifest = read(root / 'MANIFEST-v2.json')
    for name, expected in manifest['files'].items():
        assert sha(root / name) == expected, name
    tasks = read(root / 'schedule-v1.json')['rows']
    items = {r['id']: r for r in read(root / 'grading-items-v1.json')['rows']}
    assert len(items) == 24 and len(tasks) == 96
    assert {(r['prompt_id'], r['arm']) for r in tasks} == {(i, a) for i in items for a in ARMS}
    result = read(root / 'mirror-v2/result.json')
    assert result['status'] == 'completed_fresh_event_comparison'
    assert result['attempted_generations'] == result['completed_generations'] == 96
    assert result['manifest_sha256'] == sha(root / 'MANIFEST-v2.json')
    records = {r['file']: r['sha256'] for r in result['records']}
    assert len(records) == 96
    answers = {}
    for task in tasks:
        name = 'answer-' + task['case_id'] + '.json'
        path = root / 'mirror-v2' / name
        assert sha(path) == records[name]
        a = read(path)
        assert a['task'] == task and a['manifest_sha256'] == result['manifest_sha256']
        assert hashlib.sha256(a['generation']['response'].encode()).hexdigest() == a['answer_text_sha256']
        answers[task['case_id']] = a
    audits = [read(f) for f in (root / 'audits-v2').glob('audit-*.json')]
    assert any(a['complete'] and a['checked_answers'] == 96 and a['record_hashes'] == records for a in audits), 'Complete execution audit required'
    return items, answers


def packet(root, output):
    items, answers = load_run(root)
    facts = read(root / 'grader-facts-v1.json')
    output.mkdir(exist_ok=False)
    rows, links = [], []
    for number, (cid, answer) in enumerate(answers.items(), 1):
        item = items[answer['task']['prompt_id']]
        g = answer['generation']
        key = answer_key(answer)
        row = {'case_id': cid, 'answer_identity_sha256': key,
               'full_answer_read': False, 'reviewer': None, 'rationale': None,
               'event_identity': None, 'view_difference_rationale': None,
               'required_facts': {f: {'label': None, 'rationale': None} for f in item['required_fact_ids']}}
        row.update({f: None for f in VIEWS + FLAGS})
        rows.append(row)
        filename = '%03d.html' % number
        reference = {f: facts['facts'][f] for f in item['required_fact_ids']}
        document = ('<!doctype html><meta charset="utf-8"><title>Manual answer review</title>'
                    '<style>body{max-width:1000px;margin:2rem auto;font:18px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
                    '<p>Primary review; not independent or fully blind. Read the complete answer.</p>'
                    '<h1>' + html.escape(cid) + '</h1><h2>Prompt</h2><pre>' + html.escape(item['prompt']) +
                    '</pre><h2>Complete answer</h2><pre>' + html.escape(g['response']) +
                    '</pre><h2>Registered facts</h2><pre>' + html.escape(json.dumps(reference, indent=2, ensure_ascii=False)) +
                    '</pre><p>Stop reason: ' + html.escape(str(g['stop_reason'])) + '</p><p>Identity: ' + key + '</p>')
        (output / filename).write_text(document)
        links.append('<li><a href="' + filename + '">' + html.escape(cid) + '</a></li>')
    write(output / 'review-template.json', {'review_method': 'primary_manual_not_independent_not_fully_blind', 'rows': rows})
    write(output / 'packet-provenance.json', {'manifest_sha256': sha(root / 'MANIFEST-v2.json'),
                                            'script_sha256': sha(Path(__file__)), 'answers': len(rows),
                                            'files': {f.name: sha(f) for f in sorted(output.iterdir())}})
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Full-answer review: 96 answers</h1><ol>' + ''.join(links) + '</ol>')


def validate_review(review, answers, items):
    assert review['review_method'] == 'primary_manual_not_independent_not_fully_blind'
    rows = review['rows']
    assert len(rows) == len(answers) and len({r['case_id'] for r in rows}) == len(rows)
    assert {r['case_id'] for r in rows} == set(answers)
    for r in rows:
        a = answers[r['case_id']]
        item = items[a['task']['prompt_id']]
        assert r['answer_identity_sha256'] == answer_key(a), 'Review identity mismatch'
        assert r['full_answer_read'] is True and isinstance(r['reviewer'], str) and r['reviewer'].strip()
        assert isinstance(r['rationale'], str) and r['rationale'].strip()
        assert r['event_identity'] in ('matched', 'mismatched', 'unresolved', 'not_applicable')
        for f in VIEWS:
            assert r[f] in LABELS
        for f in FLAGS:
            assert r[f] is None or type(r[f]) is bool
        assert set(r['required_facts']) == set(item['required_fact_ids'])
        for f in r['required_facts'].values():
            assert f['label'] in FACT_LABELS and isinstance(f['rationale'], str) and f['rationale'].strip()
        if r[VIEWS[0]] == 'correct':
            assert all(f['label'] == 'acknowledged' for f in r['required_facts'].values())
            assert r['event_identity'] in ('matched', 'not_applicable')
            assert r['false_central_denial'] is False
        if r[VIEWS[0]] != r[VIEWS[1]]:
            assert isinstance(r['view_difference_rationale'], str) and r['view_difference_rationale'].strip()
    return {r['case_id']: r for r in rows}


def possible(label):
    return (0, 1) if label == 'unresolved' else (int(label == 'correct'),)


def paired(items, native, intervention, view, selected):
    gains, losses, uncertain, gain_topics, possible_topics = [], [], [], set(), set()
    low_net = high_net = low_gains = high_gains = low_losses = high_losses = 0
    for pid in selected:
        n, i = native[pid][view], intervention[pid][view]
        pairs = [(x, y) for x in possible(n) for y in possible(i)]
        dg = [int(x == 0 and y == 1) for x, y in pairs]
        dl = [int(x == 1 and y == 0) for x, y in pairs]
        dn = [y - x for x, y in pairs]
        low_net += min(dn); high_net += max(dn)
        low_gains += min(dg); high_gains += max(dg)
        low_losses += min(dl); high_losses += max(dl)
        if min(dg): gains.append(pid); gain_topics.add(items[pid]['topic'])
        if max(dg): possible_topics.add(items[pid]['topic'])
        if min(dl): losses.append(pid)
        if 'unresolved' in (n, i): uncertain.append(pid)
    return {'gains': gains, 'losses': losses, 'unresolved_pairs': uncertain,
            'gain_event_clusters': sorted(gain_topics),
            'possible_gain_event_clusters': sorted(possible_topics),
            'net_gain_bounds': [low_net, high_net], 'gain_bounds': [low_gains, high_gains],
            'loss_bounds': [low_losses, high_losses]}


def criterion(sensitive, controls, rules):
    r = rules['practical_transfer_support']
    met = (sensitive['gain_bounds'][0] >= r['at_least_newly_correct_sensitive_answers'] and
           len(sensitive['gain_event_clusters']) >= r['at_least_gain_event_clusters'] and
           sensitive['net_gain_bounds'][0] > 0 and
           controls['loss_bounds'][1] <= r['native_correct_control_losses_allowed'])
    impossible = (sensitive['gain_bounds'][1] < r['at_least_newly_correct_sensitive_answers'] or
                  len(sensitive['possible_gain_event_clusters']) < r['at_least_gain_event_clusters'] or
                  sensitive['net_gain_bounds'][1] <= 0 or
                  controls['loss_bounds'][0] > r['native_correct_control_losses_allowed'])
    return 'met' if met else 'not_met' if impossible else 'unresolved'


def analyze(items, answers, review, rules):
    rows = validate_review(review, answers, items)
    arms = {arm: {a['task']['prompt_id']: rows[cid] for cid, a in answers.items() if a['task']['arm'] == arm} for arm in ARMS}
    assert all(set(v) == set(items) for v in arms.values())
    selections = {g: [i for i, r in items.items() if r['group'] == g] for g in ('sensitive', 'ordinary', 'uncertainty')}
    selections['controls'] = selections['ordinary'] + selections['uncertainty']
    selections['false_premises'] = [i for i, r in items.items() if r['group'] == 'sensitive' and r['form'] == 'inaccurate_premise']
    for form in ('open', 'direct', 'accurate_premise', 'inaccurate_premise'):
        selections['form:' + form] = [i for i, r in items.items() if r['group'] == 'sensitive' and r['form'] == form]
    for topic in sorted({r['topic'] for r in items.values() if r['group'] == 'sensitive'}):
        selections['event:' + topic] = [i for i, r in items.items() if r['topic'] == topic]
    result = {'scope': 'Fresh panel only; four event clusters, not sixteen independent events.',
              'review_method': review['review_method'], 'answers_reviewed': len(rows),
              'native_mechanism_established': False, 'general_quality_preserved_proven': False,
              'goal_complete': False, 'scores': {}, 'paired_vs_native': {}, 'flags': {}, 'premise_pairs': {}}
    for arm, values in arms.items():
        result['scores'][arm] = {view: {name: dict(collections.Counter(values[p][view] for p in ids)) for name, ids in selections.items()} for view in VIEWS}
        result['flags'][arm] = {flag: {name: dict(collections.Counter('unresolved_or_not_applicable' if values[p][flag] is None else str(values[p][flag]).lower() for p in ids)) for name, ids in selections.items()} for flag in FLAGS}
        result['premise_pairs'][arm] = {topic: {form: {view: values[p][view] for view in VIEWS}
            for p, item in items.items() if item['topic'] == topic
            for form in [item['form']] if form in ('accurate_premise', 'inaccurate_premise')}
            for topic in sorted({r['topic'] for r in items.values() if r['group'] == 'sensitive'})}
        if arm != 'native':
            result['paired_vs_native'][arm] = {view: {name: paired(items, arms['native'], values, view, ids) for name, ids in selections.items()} for view in VIEWS}
    primary = result['paired_vs_native'][rules['primary_arm']][VIEWS[0]]
    result['registered_primary_criterion'] = criterion(primary['sensitive'], primary['controls'], rules)
    result['native_sensitive_correct'] = result['scores']['native'][VIEWS[0]]['sensitive'].get('correct', 0)
    result['native_sensitive_ceiling'] = result['native_sensitive_correct'] == len(selections['sensitive'])
    result['paired_detail'] = {pid: {arm: {k: arms[arm][pid][k] for k in VIEWS + FLAGS + ('rationale', 'required_facts', 'event_identity', 'view_difference_rationale')} for arm in ARMS} for pid in items}
    return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'packet':
        packet(P, Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == 'analyze':
        items, answers = load_run(P)
        review_path = Path(sys.argv[2])
        result = analyze(items, answers, read(review_path), read(P / 'decision-rules-v1.json'))
        result['provenance'] = {'manifest_sha256': sha(P / 'MANIFEST-v2.json'),
                                'review_sha256': sha(review_path), 'script_sha256': sha(Path(__file__)),
                                'rules_sha256': sha(P / 'decision-rules-v1.json')}
        write(Path(sys.argv[3]), result)
        print(json.dumps({k: result[k] for k in ('answers_reviewed', 'registered_primary_criterion', 'native_sensitive_ceiling')}))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
