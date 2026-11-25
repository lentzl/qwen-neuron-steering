"""Remote J32: two exact replays then72 independent signed event-confirmation answers."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

P = Path(__file__).resolve().parent
assert str(P).startswith('REMOTE_HOME/interpretability/'), 'Model work is remote only'
OUT = P/'run-v1'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXACT = ['prompt', 'input_token_ids', 'formatted_input', 'generated_token_ids', 'response',
         'identity', 'seed', 'temperature', 'max_new_tokens', 'stop_reason']


def atomic(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    temp = path.with_name(path.name+'.writing')
    with temp.open('wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def main():
    started = time.monotonic(); OUT.mkdir(exist_ok=False)
    process = {'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'automatic_retry': False}
    counts = {'generations': 0, 'feature_prefills': 0, 'optimizer_updates': 0}
    caps = {'generations': 74, 'feature_prefills': 0, 'optimizer_updates': 0}
    records, replays = [], {}; stage = 'validation'
    def save(name, value):
        assert not (OUT/name).exists(), name
        atomic(OUT/name, value)
    def status(label):
        nonlocal stage
        stage = label
        v = {'status': 'running', 'stage': stage, 'process': process, 'counts': dict(counts),
            'completed_generations': len(records), 'elapsed_seconds': time.monotonic()-started}
        atomic(OUT/'status.json', v); print(json.dumps(v), flush=True)
    def deadline(sig, frame): raise TimeoutError('Fixed120-minute cap including load; no retry or extension')
    save('process.json', process)
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 7200-(time.monotonic()-started)))
    try:
        manifest = read(P/'MANIFEST-v1.json')
        assert manifest['status'] == 'frozen_for_execution'
        assert manifest['budget'] == dict(caps, wall_seconds_including_load=7200, max_new_tokens=2048,
            automatic_retry=False, automatic_extension=False)
        for name, expected in manifest['files'].items(): assert sha(P/name) == expected, name
        for name, expected in manifest['remote_files'].items(): assert sha(name) == expected, name
        assert process['boot_id'] == manifest['boot_id']
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        assert shutil.disk_usage(P).free > 8*1024**3
        core = read(P/'core-source-contract-v1.json')
        tasks = read(P/'schedule-v1.json')['rows']; assert len(tasks) == 74
        prompts = read(P/'model-prompts-v1.json')['rows']; assert len(prompts) == 24
        refs = read(P/'core-replay-references-v1.json')['references']
        assert set(refs) == {'native', 'positive'}
        assert [r['phase'] for r in tasks] == ['replay']*2+['scored']*72
        assert {(r['prompt_id'], r['arm']) for r in tasks[2:]} == {(r['id'], a) for r in prompts for a in ['native', 'positive', 'negative']}
        assert len({r['case_id'] for r in tasks}) == 74
        status('loading')
        import torch as t
        t.set_num_threads(1); t.set_num_interop_threads(1); t.manual_seed(1729); t.cuda.manual_seed_all(1729)
        sys.path.insert(0, core['runtime_directory'])
        from v2_backend import Session
        from causal_patch import FixedMLPOffset, fingerprint
        session = Session(core['model_snapshot'], OUT/'runtime.json')
        assert session.identity == core['model_identity']
        runtime = read(OUT/'runtime.json')
        for key, value in core['runtime_metadata'].items(): assert runtime[key] == value, key
        session.model.requires_grad_(False)
        versions = {n: x._version for n, x in session.model.named_parameters()}
        def intact():
            assert not session.model.training and not session.model.is_gradient_checkpointing
            assert versions == {n: x._version for n, x in session.model.named_parameters()}
            assert all(not x.requires_grad and x.grad is None for x in session.model.parameters())
            assert all(not x._forward_hooks and not x._forward_pre_hooks for x in session.model.modules())
        offsets = t.load(core['offsets_path'], map_location='cpu', weights_only=True)['variants']['mlp__complement']
        contract = read(core['contract_path'])['mlp__complement']
        def offset_proof():
            return {str(l): {'selected_neurons': v['indices'].numel(),
                'coefficient_fingerprints': {k: fingerprint(t, z) for k, z in v.items()}} for l, v in offsets.items()}
        assert offset_proof() == contract
        candidate = read(P/'selected-candidate-v1.json')
        assert candidate['positive'] == core['configurations']['positive']
        assert candidate['negative'] == core['configurations']['negative']
        assert core['configurations']['negative']['alpha'] == -core['configurations']['positive']['alpha']
        assert read(P/'signed-operator-test-v1.json')['passed']
        rows = {r['id']: r for r in prompts}; references = {}; serial = {}
        for arm, ref in refs.items():
            cid = 'replay_core__'+arm
            rows[cid] = {'id': cid, 'prompt': ref['answer']['generation']['prompt']}
            references[cid] = ref['answer']
        for cid, row in rows.items():
            ids, text = session.encode(row)
            expected = session.tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert text == expected and len(ids) > 3
            serial[cid] = {'input_token_ids': ids, 'formatted_input': text}
            if cid in references: assert all(serial[cid][k] == references[cid]['generation'][k] for k in serial[cid])
        save('serialized-prompts-v1.json', serial); intact()
        for index, task in enumerate(tasks):
            if index >= 2: assert len(replays) == 2 and all(replays.values())
            cid, arm = task['prompt_id'], task['arm']; ids = serial[cid]['input_token_ids']; row = rows[cid]
            assert arm in ['native', 'positive', 'negative']
            patch = None if arm == 'native' else FixedMLPOffset(t, session.model, session.blocks, ids, offsets, core['configurations'][arm]['alpha'])
            first = []
            def inspect_first(module, args, kwargs):
                if first: return
                x = kwargs.get('input_ids', args[0] if args else None)
                assert x is not None and x.shape == (1, len(ids)) and x[0].tolist() == ids
                cache = kwargs.get('past_key_values')
                assert cache is None or int(cache.get_seq_length()) == 0
                first.append({'full_prompt_exact': True, 'initial_cache_empty': True,
                    'cache_provided': cache is not None, 'prompt_tokens': len(ids)})
            assert counts['generations'] < 74 and time.monotonic()-started < 7200
            assert shutil.disk_usage(P).free > 8*1024**3
            handle = session.model.register_forward_pre_hook(inspect_first, with_kwargs=True)
            counts['generations'] += 1; status(task['case_id'])
            try:
                with patch if patch is not None else nullcontext():
                    g = session.generate({'case_id': task['case_id'], 'prompt': row['prompt']},
                        seed=1729, temperature=0, max_new_tokens=2048)
            finally: handle.remove()
            proof = {'intervention': False} if patch is None else patch.verify()
            # Compare serialized forms: layer keys are integers in live Python and strings in archived JSON.
            proof = json.loads(json.dumps(proof, allow_nan=False))
            assert len(first) == 1 and g['identity'] == core['model_identity']
            assert g['input_token_ids'] == ids and g['formatted_input'] == serial[cid]['formatted_input']
            assert g['prompt'] == row['prompt'] and (g['seed'], g['temperature'], g['max_new_tokens']) == (1729, 0, 2048)
            assert g['generated_tokens'] == len(g['generated_token_ids']) <= 2048
            assert g['response'] == session.tokenizer.decode(g['generated_token_ids'], skip_special_tokens=True)
            assert g['hook_calls'] == g['strength'] == 0 and g['layer'] is None
            if patch is not None:
                assert proof['neurons'] == 892 and proof['decode_hook_calls'] == 0
                assert proof['alpha'] == core['configurations'][arm]['alpha']
                for layer, value in proof['records'].items():
                    assert value['positions'] == list(range(3, len(ids)))
                    assert value['coefficient_fingerprints'] == contract[layer]['coefficient_fingerprints']
            intact()
            name = 'answer-'+task['case_id']+'.json'
            save(name, {'task': task, 'generation': g, 'intervention_proof': proof,
                'independent_prompt_proof': first[0], 'manifest_sha256': sha(P/'MANIFEST-v1.json'),
                'paired_truthful_prompt_used': False, 'conditional_coefficients': None})
            records.append(dict(task, file=name, sha256=sha(OUT/name)))
            if task['phase'] == 'replay':
                ref = references[cid]
                assert all(g[k] == ref['generation'][k] for k in EXACT)
                assert proof == ref['intervention_proof']
                replays[task['case_id']] = True
                save('replay-'+task['case_id']+'.json', {'full_generation_exact': True, 'operator_audit_exact': True})
        assert counts == caps and len(records) == 74 and offset_proof() == contract
        intact()
        for name, expected in manifest['remote_files'].items(): assert sha(name) == expected, name
        result = {'status': 'completed_signed_event_confirmation', 'process': process, 'records': records,
            'artifacts': [], 'counts': counts, 'caps': caps, 'replay_gates': replays, 'conditional_enabled': False,
            'matrix_complete': True, 'base_unchanged': True, 'offsets_unchanged': True, 'controller_unchanged': None,
            'source_files_unchanged': True, 'all_prompts_independent': True,
            'elapsed_seconds': time.monotonic()-started, 'manifest_sha256': sha(P/'MANIFEST-v1.json'),
            'review_pending': True, 'goal_complete': False, 'automatic_retry': False}
        save('result.json', result); atomic(OUT/'status.json', dict(result, records=None))
    except BaseException as exc:
        failure = {'status': 'wall_cap_reached' if isinstance(exc, TimeoutError) else 'failed',
            'error': repr(exc), 'traceback': traceback.format_exc(), 'process': process, 'stage': stage,
            'counts': counts, 'completed_generations': len(records), 'records': records, 'replay_gates': replays,
            'elapsed_seconds': time.monotonic()-started, 'automatic_retry': False}
        atomic(OUT/'failure.json', failure); atomic(OUT/'status.json', dict(failure, records=None)); raise
    finally: signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__': main()
