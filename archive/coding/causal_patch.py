"""Fixed MLP activation offsets, prefill only; no donor prompt at inference."""
import math
from writer_operands import fingerprint
class FixedMLPOffset:
    def __init__(self,t,model,blocks,ids,offsets,alpha):
        self.t=t;self.model=model;self.blocks=blocks;self.ids=list(ids);self.offsets=offsets;self.alpha=alpha
        assert isinstance(alpha,(int,float)) and math.isfinite(alpha)
        assert len(ids)>3 and offsets
        self.layers=sorted(offsets);self.records={};self.handles=[];self.calls={l:0 for l in self.layers}
        self.finished=False;self.prompt_calls=0;self.entered=False;self.coefficients={}
        for l,v in offsets.items():
            assert type(l)is int and 0<=l<len(blocks)
            assert set(v)=={"indices","delta"}
            i,x=v["indices"],v["delta"]
            assert i.dtype==t.long and i.ndim==1 and i.numel()>0 and i.unique().numel()==i.numel()
            assert int(i.min())>=0 and int(i.max())<17408
            assert x.dtype==t.float32 and x.shape==i.shape and bool(t.isfinite(x).all())
            self.coefficients[l]={k:fingerprint(t,z) for k,z in v.items()}
    def __enter__(self):
        assert not self.entered and not self.finished
        self.entered=True
        def before(module,args,kwargs):
            ids=kwargs.get("input_ids",args[0] if args else None)
            assert self.prompt_calls==0 and ids is not None and ids.tolist()==[self.ids]
            self.prompt_calls+=1
        def after(module,args,out):
            assert self.prompt_calls==1 and all(v==1 for v in self.calls.values())
            self.finished=True
            for h in self.handles:h.remove()
            self.handles=[]
        self.handles.append(self.model.register_forward_pre_hook(before,with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(after))
        for l in self.layers:
            def patch(module,args,layer=l):
                assert self.prompt_calls==1 and not self.finished and self.calls[layer]==0
                self.calls[layer]+=1;t=self.t;x=args[0]
                assert x.dtype==t.bfloat16 and tuple(x.shape)==(1,len(self.ids),17408)
                coef=self.offsets[layer]
                assert {k:fingerprint(t,z) for k,z in coef.items()}==self.coefficients[layer]
                indices=coef["indices"].to(x.device);delta=coef["delta"].to(x.device)
                before=fingerprint(t,x)
                if self.alpha==0:y=x
                else:
                    y=x.clone()
                    y[:,3:,indices]=(x[:,3:,indices].float()+float(self.alpha)*delta).to(x.dtype)
                other=t.ones(17408,dtype=t.bool,device=x.device);other[indices]=False
                assert t.equal(x[:,:3],y[:,:3]) and t.equal(x[:,:,other],y[:,:,other])
                assert fingerprint(t,x)==before
                actual=(y[:,3:,indices].float()-x[:,3:,indices].float())
                requested=(float(self.alpha)*delta).expand_as(actual)
                self.records[layer]=dict(layer=layer,selected_neurons=indices.numel(),positions=list(range(3,len(self.ids))),
                    alpha=float(self.alpha),before=before,after=fingerprint(t,y),coefficient_fingerprints=self.coefficients[layer],
                    input_unmodified=True,unselected_coordinates_exact=True,earlier_positions_exact=True,
                    zero_is_object_identity=self.alpha==0 and y is x,changed_elements=int(t.count_nonzero(actual)),
                    requested_l2=float(requested.double().norm()),realized_l2=float(actual.double().norm()),
                    rounding_l2=float((actual.double()-requested.double()).norm()))
                return None if self.alpha==0 else (y,)+args[1:]
            self.handles.append(self.blocks[l].mlp.down_proj.register_forward_pre_hook(patch))
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles=[];self.entered=False
        return False
    def verify(self):
        assert self.finished and not self.entered and not self.handles and self.prompt_calls==1
        assert set(self.records)==set(self.layers) and all(v==1 for v in self.calls.values())
        for l,r in self.records.items():
            assert r["input_unmodified"] and r["unselected_coordinates_exact"] and r["earlier_positions_exact"]
            expected=abs(self.alpha)*float(self.offsets[l]["delta"].double().norm())*math.sqrt(len(self.ids)-3)
            assert math.isclose(r["requested_l2"],expected,rel_tol=2e-6,abs_tol=1e-7)
            assert 0<=r["changed_elements"]<=r["selected_neurons"]*(len(self.ids)-3)
            if self.alpha==0:assert r["zero_is_object_identity"] and r["changed_elements"]==r["realized_l2"]==0 and r["before"]==r["after"]
        return dict(passed=True,alpha=self.alpha,layers=self.layers,neurons=sum(v["indices"].numel() for v in self.offsets.values()),
            positions_start=3,prompt_calls=1,decode_hook_calls=0,hooks_removed_at_end_of_prefill=True,
            paired_truthful_prompts_at_inference=0,records=self.records)
