"""Verify the retained candidate-to-intervention lineage on CPU; no model calls."""
from pathlib import Path
import hashlib,json,math
import torch

ROOT=Path(__file__).resolve().parents[1]
def load(name):return torch.load(ROOT/'artifacts/discovery'/name,map_location='cpu',weights_only=True)
def equal(a,b):
    assert set(a)==set(b)
    for layer in a:
        assert set(a[layer])==set(b[layer])
        for k in a[layer]:assert torch.equal(a[layer][k],b[layer][k]),(layer,k)

def main():
    torch.set_num_threads(1)
    for r in json.loads((ROOT/'provenance/discovery-recovery.json').read_text())['files']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    c=load('candidates.pt');causal=load('causal-offsets.pt')['variants'];quality=load('quality-offsets.pt')
    assert c['layer'].shape==c['neuron'].shape==c['fixed_delta'].shape==(4096,)
    assert torch.equal(c['fixed_delta'],(c['mean_deltas']['fibre']+c['mean_deltas']['cnc'])/2)
    assert bool((c['scores']['fibre']>0).all()) and bool((c['scores']['cnc']>0).all())
    flat=c['layer']*17408+c['neuron']
    assert len(set(flat.tolist()))==4096
    assert list(range(4096))==sorted(range(4096),key=lambda i:(-float(c['joint_score'][i]),int(flat[i])))
    assert c['coefficient_sha256']==hashlib.sha256((ROOT/'artifacts/discovery/writer-coefficients.pt').read_bytes()).hexdigest()
    for n in [256,1024,4096]:
        built={}
        for layer in sorted(c['layer'][:n].unique().tolist()):
            mask=c['layer'][:n]==layer
            built[layer]={'indices':c['neuron'][:n][mask],'delta':c['fixed_delta'][:n][mask]}
        equal(built,causal[f'ranked_{n}'])
    equal(causal['ranked_1024'],quality['mlp'])
    gen=torch.Generator().manual_seed(1730);randoms={}
    for layer,v in sorted(quality['mlp'].items()):
        available=torch.ones(17408,dtype=torch.bool);available[causal['ranked_4096'][layer]['indices']]=False
        pool=torch.where(available)[0];indices=pool[torch.randperm(len(pool),generator=gen)[:len(v['indices'])]]
        randoms[layer]={'indices':indices,'delta':v['delta']}
    equal(randoms,quality['random'])
    final=torch.load(ROOT/'artifacts/offset-partitions.pt',map_location='cpu',weights_only=True)['variants']
    for arm in ['mlp','random']:
        for label,layers in [('full',set(quality[arm])),('layers19_20',{19,20}),('complement',set(quality[arm])-{19,20})]:
            equal({l:quality[arm][l] for l in layers},final[f'{arm}__{label}'])
    def norm(x):return math.sqrt(sum(float(v['delta'].double().square().sum()) for v in x.values()))
    alpha=norm(final['mlp__full'])/norm(final['mlp__complement'])
    assert math.isclose(alpha,2.484356321608939,rel_tol=1e-14)
    assert sum(v['indices'].numel() for v in final['mlp__complement'].values())==892
    print('PASS: reference hash; saved top4096 ordering/deltas; 256/1024/4096 packaging; random control; all partitions; 892 coefficients and dose.')
    print('Scope: validates the retained candidate-to-offset lineage, not regeneration of gradients or ranking over the full unsaved pool.')

if __name__=='__main__':main()
