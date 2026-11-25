"""Portable entry point using the study's original BF16 inference backend and hook."""
from pathlib import Path
from contextlib import nullcontext
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent/'runtime'))
ALPHA = 2.484356321608939
ARTIFACT_SHA = '6aefb83dae922e546c3fcdc14bb372e2613cd3b020308bc45dd0936b212ddd79'
CONFIG_FILE_SHA = '69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True,help='HF snapshot directory named with the pinned revision hash')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--cases',type=Path,help='JSON list (or object with rows) of prompts/cases')
    p.add_argument('--out',type=Path,required=True,help='New output directory; never overwrites saved evidence')
    p.add_argument('--arms',nargs='+',choices=['native','positive','negative'],default=['native','positive'])
    p.add_argument('--max-new-tokens',type=int,default=2048)
    a=p.parse_args()
    import torch
    from v2_backend import Session
    from causal_patch import FixedMLPOffset, fingerprint
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
    artifact=ROOT/'artifacts/offset-partitions.pt'
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==ARTIFACT_SHA
    offsets=torch.load(artifact,map_location='cpu',weights_only=True)['variants']['mlp__complement']
    contract=json.loads((ROOT/'artifacts/offset-contract.json').read_text())['mlp__complement']
    proof={str(l):{'selected_neurons':v['indices'].numel(),'coefficient_fingerprints':{k:fingerprint(torch,z) for k,z in v.items()}} for l,v in offsets.items()}
    assert proof==contract
    cases=[{'id':'custom','prompt':a.prompt}] if a.prompt is not None else json.loads(a.cases.read_text())
    if isinstance(cases,dict):cases=cases['rows']
    assert isinstance(cases,list) and cases
    a.out.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256((a.snapshot/'config.json').read_bytes()).hexdigest()==CONFIG_FILE_SHA
    session=Session(a.snapshot,a.out/'runtime.json')
    expected=json.loads((ROOT/'evidence/engagement/reserved/core-source-contract-v1.json').read_text())
    # HF includes the installation directory in config.to_dict()['_name_or_path'].
    # Validate exact pinned config bytes above, rather than require the old private path.
    assert {k:v for k,v in session.identity.items() if k!='config_sha256'}=={k:v for k,v in expected['model_identity'].items() if k!='config_sha256'}
    runtime=json.loads((a.out/'runtime.json').read_text())
    # Record rather than conceal any difference in the original observed runtime.
    differences={k:{'expected':v,'actual':runtime.get(k)} for k,v in expected['runtime_metadata'].items() if runtime.get(k)!=v}
    (a.out/'runtime-comparison.json').write_text(json.dumps(differences,indent=2)+'\n')
    session.model.requires_grad_(False)
    versions={n:x._version for n,x in session.model.named_parameters()}
    for i,case in enumerate(cases):
        ids,_=session.encode(case)
        for arm in a.arms:
            patch=None if arm=='native' else FixedMLPOffset(torch,session.model,session.blocks,ids,offsets,ALPHA if arm=='positive' else -ALPHA)
            with patch if patch is not None else nullcontext():
                g=session.generate(dict(case,case_id=str(case.get('id',i))+'__'+arm),seed=1729,temperature=0,max_new_tokens=a.max_new_tokens)
            result={'arm':arm,'generation':g,'intervention_proof':{'intervention':False} if patch is None else patch.verify()}
            assert versions=={n:x._version for n,x in session.model.named_parameters()}
            assert all(not x._forward_hooks and not x._forward_pre_hooks for x in session.model.modules())
            (a.out/f'{i:03d}-{arm}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'case':i,'arm':arm,'response':g['response']}),flush=True)

if __name__=='__main__':main()
