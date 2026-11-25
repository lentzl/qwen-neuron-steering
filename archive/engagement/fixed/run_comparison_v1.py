"""Remote-only, fixed three-arm comparison; 144 calls, no training or retries."""
from pathlib import Path
from contextlib import nullcontext
import copy
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
assert str(P).startswith('REMOTE_HOME/interpretability/'), 'Remote model work only'
OUT = P/'run-v1'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path, obj):
    data = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()
    tmp = path.with_name(path.name+'.writing')
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    started = time.monotonic(); epoch = time.time()
    manifest = read(P/'MANIFEST-v1.json')
    for name, digest in manifest['files'].items(): assert sha(P/name) == digest, name
    for name, digest in manifest['remote_files'].items(): assert sha(name) == digest, name
    prompts = read(P/'model-prompts-v1.json')['rows']
    schedule = read(P/'schedule-v1.json')['rows']
    configs = manifest['configurations']
    by = {p['id']: p for p in prompts}
    assert len(by) == len(prompts) == 48 and len(schedule) == 144
    assert all(set(p) == {'id', 'prompt'} for p in prompts)
    assert len({r['case_id'] for r in schedule}) == 144
    assert {(r['prompt_id'], r['arm']) for r in schedule} == {(p, a) for p in by for a in configs}
    assert len(configs) == 3
    for r in schedule: assert r['configuration_id'] == configs[r['arm']]['configuration_id']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert shutil.disk_usage(P).free > 8*1024**3
    OUT.mkdir(exist_ok=False)
    process = {'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
               'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'automatic_retry': False}
    atomic(OUT/'process.json', process)
    attempts = 0; records = []; stage = 'validation'
    atomic(OUT/'budget-start-v1.json', {'started_epoch': epoch, 'deadline_epoch': epoch+10800,
                                      'max_generations': 144, 'max_new_tokens': 2048, 'includes_loading_and_validation': True})
    def status(label):
        nonlocal stage
        stage = label
        value = {'status': 'running', 'stage': stage, 'process': process, 'attempted_generations': attempts,
                 'completed_generations': len(records), 'elapsed_seconds': time.monotonic()-started}
        atomic(OUT/'status.json', value); print(json.dumps(value), flush=True)
    def deadline(sig, frame): raise TimeoutError('Frozen three-hour cap; no automatic retry or extension')
    signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, max(.001,10800-(time.monotonic()-started)))
    try:
        status('loading')
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
        sys.path.insert(0, manifest['runtime_directory'])
        from v2_backend import Session
        from causal_patch import FixedMLPOffset, fingerprint
        variants = torch.load(manifest['offsets_path'], map_location='cpu', weights_only=True)['variants']
        contract = read(manifest['contract_path'])
        rebuilt = {name: {str(layer): {'selected_neurons': v['indices'].numel(),
                    'coefficient_fingerprints': {k: fingerprint(torch,z) for k,z in v.items()}}
                    for layer,v in values.items()} for name,values in variants.items()}
        assert rebuilt == contract
        session = Session(manifest['model_snapshot'], OUT/'runtime.json')
        assert session.identity == manifest['model_identity']
        current_runtime = read(OUT/'runtime.json')
        for key, expected in manifest['runtime_metadata'].items():
            assert current_runtime[key] == expected, key
        model = session.model; model.requires_grad_(False)
        versions = {name: value._version for name,value in model.named_parameters()}
        def intact():
            assert not model.training
            assert versions == {name: value._version for name,value in model.named_parameters()}
            assert all(value.grad is None for value in model.parameters())
            assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
        intact()
        config = copy.deepcopy(model.generation_config)
        config.do_sample = False; config.temperature = 1.; config.top_p = 1.; config.top_k = 0
        config.max_new_tokens = 2048; config.num_beams = 1; config.num_return_sequences = 1
        config.repetition_penalty = 1.; config.pad_token_id = session.tokenizer.pad_token_id
        assert config.forced_bos_token_id is None and config.forced_eos_token_id is None
        atomic(OUT/'effective-generation-config-v1.json', config.to_dict())
        serialized = {}
        for row in prompts:
            ids, text = session.encode(row)
            assert len(ids) > 3
            serialized[row['id']] = {'input_token_ids': ids, 'formatted_input': text}
        atomic(OUT/'serialized-prompts-v1.json', serialized)
        atomic(OUT/'runtime-validation-v1.json', {'passed': True, 'coefficient_contract_rebuilt': True,
              'model_identity_matched': True, 'extra_forwards': 0, 'replay_generations': 0,
              'replay_limit': 'No new exact replay experiment; historical exact replays and hash-pinned backend retained. Each new intervention receives per-layer application proof.'})
        for task in schedule:
            assert attempts < 144 and time.monotonic()-started < 10800
            assert shutil.disk_usage(P).free > 8*1024**3
            row = by[task['prompt_id']]; configuration = configs[task['arm']]
            ids = serialized[row['id']]['input_token_ids']
            if task['arm'] == 'native': patch = None
            else:
                offsets = variants['mlp__'+configuration['partition']]
                patch = FixedMLPOffset(torch, model, session.blocks, ids, offsets, configuration['alpha'])
            attempts += 1; status(task['case_id'])
            with patch if patch is not None else nullcontext():
                generation = session.generate({'case_id': task['case_id'], 'prompt': row['prompt']},
                                              seed=1729, temperature=0, max_new_tokens=2048)
            proof = {'intervention': False} if patch is None else patch.verify()
            if patch is not None:
                count = 1024 if task['arm'] == 'full1024_plus2' else 892
                layers = list(range(22)) if count == 1024 else [l for l in range(22) if l not in (19,20)]
                assert proof['neurons'] == count and proof['layers'] == layers
                assert proof['alpha'] == configuration['alpha'] and proof['decode_hook_calls'] == 0
                for layer, record in proof['records'].items():
                    expected = contract['mlp__'+configuration['partition']][str(layer)]
                    assert record['coefficient_fingerprints'] == expected['coefficient_fingerprints']
                    assert record['positions'] == list(range(3,len(ids)))
            assert generation['identity'] == session.identity
            assert generation['input_token_ids'] == ids
            assert generation['formatted_input'] == serialized[row['id']]['formatted_input']
            assert generation['seed'] == 1729 and generation['temperature'] == 0 and generation['max_new_tokens'] == 2048
            assert generation['generated_tokens'] == len(generation['generated_token_ids']) <= 2048
            assert generation['response'] == session.tokenizer.decode(generation['generated_token_ids'], skip_special_tokens=True)
            # Session's hook fields describe its optional residual intervention, not the external MLP patch.
            assert generation['hook_calls'] == 0 and generation['strength'] == 0 and generation['layer'] is None
            intact()
            name = 'answer-'+task['case_id']+'.json'
            assert not (OUT/name).exists()
            atomic(OUT/name, {'task': task, 'generation': generation, 'mlp_patch': proof,
                             'external_patch_is_authoritative': True, 'paired_truthful_prompt_used': False,
                             'answer_sha256': hashlib.sha256(generation['response'].encode()).hexdigest(),
                             'manifest_sha256': sha(P/'MANIFEST-v1.json')})
            records.append({'file': name, 'sha256': sha(OUT/name), **task, 'stop_reason': generation['stop_reason']})
            del patch, generation, proof
        intact(); signal.setitimer(signal.ITIMER_REAL, 0)
        assert attempts == len(records) == 144
        result = {'status': 'completed_fixed_comparison', 'process': process,
                  'attempted_generations': attempts, 'completed_generations': len(records), 'records': records,
                  'matrix_complete': True, 'base_unchanged': True, 'optimizer_updates': 0,
                  'separate_forwards': 0, 'review_pending': True, 'goal_complete': False,
                  'elapsed_seconds': time.monotonic()-started, 'manifest_sha256': sha(P/'MANIFEST-v1.json')}
        atomic(OUT/'result.json', result); atomic(OUT/'status.json', dict(result, records=None))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result = {'status': 'wall_cap_reached' if isinstance(exc, TimeoutError) else 'failed',
                  'error': repr(exc), 'traceback': traceback.format_exc(), 'process': process, 'stage': stage,
                  'attempted_generations': attempts, 'completed_generations': len(records), 'records': records,
                  'matrix_complete': False, 'automatic_retry': False, 'elapsed_seconds': time.monotonic()-started}
        atomic(OUT/'failure.json', result); atomic(OUT/'status.json', dict(result, records=None))
        raise


if __name__ == '__main__': main()
