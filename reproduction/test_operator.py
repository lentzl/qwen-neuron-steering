"""CPU tests of the real archived operator and exact published coefficient fingerprints."""
from pathlib import Path
import hashlib, json, sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from causal_patch import FixedMLPOffset, fingerprint

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.mlp=torch.nn.Module();self.mlp.down_proj=torch.nn.Identity()
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([Block() for _ in range(64)])
    def forward(self,input_ids):
        x=torch.zeros((1,input_ids.shape[1],17408),dtype=torch.bfloat16)
        return [b.mlp.down_proj(x) for b in self.blocks]

def main():
    torch.set_num_threads(1)
    p=ROOT/'artifacts/offset-partitions.pt'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='6aefb83dae922e546c3fcdc14bb372e2613cd3b020308bc45dd0936b212ddd79'
    offsets=torch.load(p,map_location='cpu',weights_only=True)['variants']['mlp__complement']
    assert sorted(offsets)==list(range(19))+[21]
    assert sum(x['indices'].numel() for x in offsets.values())==892
    contract=json.loads((ROOT/'artifacts/offset-contract.json').read_text())['mlp__complement']
    assert contract=={str(l):{'selected_neurons':v['indices'].numel(),'coefficient_fingerprints':{k:fingerprint(torch,z) for k,z in v.items()}} for l,v in offsets.items()}
    model=Toy();ids=[10,11,12,13,14,15];inputs=torch.tensor([ids])
    for alpha in [0.,2.484356321608939,-2.484356321608939]:
        patch=FixedMLPOffset(torch,model,model.blocks,ids,offsets,alpha)
        with patch:
            outputs=model(input_ids=inputs)
            # A separate decode invocation must be completely untouched.
            decode=model(input_ids=torch.tensor([[16]]))
        audit=patch.verify()
        assert audit['decode_hook_calls']==0 and audit['neurons']==892
        for layer,out in enumerate(outputs):
            expected=torch.zeros_like(out)
            if layer in offsets:
                v=offsets[layer];expected[:,3:,v['indices']]=(alpha*v['delta']).to(torch.bfloat16)
            assert torch.equal(out,expected),(layer,alpha)
        assert all(torch.count_nonzero(x)==0 for x in decode)
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    try:
        with FixedMLPOffset(torch,model,model.blocks,ids,offsets,1):raise RuntimeError('test cleanup')
    except RuntimeError:pass
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    print('PASS: 892 exact coefficients; positive/negative/zero; prompt positions; decode untouched; exception cleanup.')

if __name__=='__main__':main()
