"""A single shared latent Q/K/V core; no intermediate policy supervision."""
import math
import torch
from torch import nn
from torch.nn import functional as F

D=128


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(D,D); self.k=nn.Linear(D,D); self.v=nn.Linear(D,D); self.out=nn.Linear(D,D)

    def heads(self,x): return x.reshape(x.shape[0],x.shape[1],4,32).transpose(1,2)

    def kv(self,x): return self.heads(self.k(x)),self.heads(self.v(x))

    def forward(self,q,kv,diagnostic=False):
        q=self.heads(self.q(q)); k,v=kv
        attended=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        weight=(q@k.transpose(-2,-1)/math.sqrt(32)).softmax(-1) if diagnostic else None
        return self.out(attended.transpose(1,2).reshape(q.shape[0],-1,D)),weight


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross=Attention(); self.self_attention=Attention()
        self.qnorm=nn.LayerNorm(D); self.memnorm=nn.LayerNorm(D); self.selfnorm=nn.LayerNorm(D); self.ffnorm=nn.LayerNorm(D)
        self.mlp=nn.Sequential(nn.Linear(D,512),nn.GELU(),nn.Linear(512,D))

    def memory(self,x): return self.cross.kv(self.memnorm(x))

    def forward(self,h,kv,diagnostic=False):
        c,w=self.cross(self.qnorm(h),kv,diagnostic); h=h+c
        n=self.selfnorm(h); s,_=self.self_attention(n,self.self_attention.kv(n)); h=h+s
        return h+self.mlp(self.ffnorm(h)),w


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Sequential(nn.Linear(5,D),nn.GELU(),nn.Linear(D,D))
        self.hnorm=nn.LayerNorm(D); self.qnorm=nn.LayerNorm(D); self.attention=Attention()
        self.out=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,D),nn.GELU(),nn.Linear(D,4))

    def forward(self,tokens,h):
        q=self.query(tokens); a,_=self.attention(self.qnorm(q),self.attention.kv(self.hnorm(h)))
        return self.out(q+a)


class Solver(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(5,D),nn.GELU(),nn.Linear(D,D))
        self.row=nn.Embedding(32,D); self.column=nn.Embedding(32,D)
        self.initial=nn.Parameter(torch.randn(16,D)*.02)
        self.slot=nn.Parameter(torch.randn(16,D)*.02); self.time=nn.Parameter(torch.randn(16,D)*.02)
        self.source=nn.Parameter(torch.randn(2,D)*.02)
        self.core=Core(); self.readout=Readout()
        y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
        self.register_buffer('rows',y.reshape(-1),persistent=False); self.register_buffer('columns',x.reshape(-1),persistent=False)

    def encode(self,tokens):
        return self.encoder(tokens)+self.row(self.rows)[None]+self.column(self.columns)[None]

    def history_kv(self,h,t):
        return self.core.memory(h+self.slot[None]+self.time[t][None,None]+self.source[1][None,None])

    def memory(self,xkv,history,projected,variant):
        if not history or variant=='remove': return xkv
        if variant=='normal': kvs=projected
        else:
            values=list(history)
            if variant=='zero': values=[torch.zeros_like(h) for h in values]
            elif variant=='reverse': values=list(reversed(values))
            elif variant=='shuffle': values=values[1:]+values[:1]
            elif variant=='other': values=[h.roll(1,0) for h in values]
            else: raise ValueError(variant)
            kvs=[self.history_kv(h,t) for t,h in enumerate(values)]
        return tuple(torch.cat([xkv[j]]+[kv[j] for kv in kvs],dim=2) for j in (0,1))

    def forward(self,tokens,steps=16,history_enabled=True,variant='normal',trace=False,return_states=False):
        x=self.encode(tokens); xkv=self.core.memory(x+self.source[0][None,None])
        h=self.initial[None].expand(len(tokens),-1,-1); history=[]; projected=[]; diagnostics=[]
        for t in range(steps):
            kv=self.memory(xkv,history,projected,variant) if history_enabled else xkv
            previous=h; h,w=self.core(h,kv,trace)
            if trace:
                diagnostics.append({'change':(h-previous).square().mean((1,2)).sqrt(),
                                    'input_attention':w[:,:,:,:1024].sum(-1).mean((1,2)),
                                    'history_attention':w[:,:,:,1024:].sum(-1).mean((1,2))})
            history.append(h)
            if history_enabled and t+1<steps: projected.append(self.history_kv(h,t))
        logits=self.readout(tokens,h)
        if return_states or trace: return logits,history,diagnostics,x
        return logits

    def adaptive(self,tokens,halt,stochastic=False,trace=False):
        """Compact active examples; stopped examples never execute another core call."""
        x=self.encode(tokens); xkv=self.core.memory(x+self.source[0][None,None])
        active=torch.arange(len(tokens),device=tokens.device); local=tokens
        h=self.initial[None].expand(len(tokens),-1,-1); history=[]; projected=[]
        final=torch.zeros(len(tokens),16,D,device=tokens.device)
        times=torch.zeros(len(tokens),dtype=torch.long,device=tokens.device)
        forced=torch.zeros(len(tokens),dtype=torch.bool,device=tokens.device)
        logprob=torch.zeros(len(tokens),device=tokens.device); events=[]; calls=[]
        for t in range(16):
            h,_=self.core(h,self.memory(xkv,history,projected,'normal'))
            calls.append(int(len(active)))
            halt_logits=halt(h)
            probability=halt_logits.sigmoid()
            if t==15:
                # Sample the last action too: continue is forced, halt is autonomous.
                stop=torch.distributions.Bernoulli(probs=probability).sample().bool() if stochastic else probability>=.5
                terminate=torch.ones_like(stop); forced[active]=~stop
            else:
                stop=torch.distributions.Bernoulli(probs=probability).sample().bool() if stochastic else probability>=.5
                terminate=stop
            selected=torch.where(stop,F.logsigmoid(halt_logits),F.logsigmoid(-halt_logits))
            logprob=logprob.index_add(0,active,selected)
            if trace: events.append({'t':t+1,'ids':active.detach().cpu().tolist(),'p':probability.detach().cpu().tolist(),
                                     'stop':stop.detach().cpu().tolist(),'log_probability':selected.detach().cpu().tolist()})
            ended=active[terminate]; final[ended]=h[terminate]; times[ended]=t+1
            if bool(terminate.all()): break
            keep=~terminate; active=active[keep]; h=h[keep]; local=local[keep]
            xkv=tuple(v[keep] for v in xkv)
            history=[v[keep] for v in history]+[h]
            projected=[tuple(v[keep] for v in kv) for kv in projected]+[self.history_kv(h,t)]
        # One readout for all actually chosen terminal states, never intermediate losses.
        logits=self.readout(tokens,final)
        return logits,{'steps':times,'forced':forced,'log_probability':logprob,'events':events,'active_core_batch_sizes':calls}


class Halt(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,h): return self.net(h.mean(1)).squeeze(-1)


def parameters(model): return sum(p.numel() for p in model.parameters())


def estimated_flops(steps,history=True):
    """Linear matmuls + QK/AV; multiply/add=2, excludes norms/activation/softmax."""
    n,s,d=1024,16,128
    encoder=2*n*(5*d+d*d)
    encoder+=2*n*2*d*d  # input K/V once, shared without detach
    head=2*n*(5*d+d*d)+2*n*d*d+2*s*2*d*d+4*n*s*d+2*n*d*d+2*n*(d*d+d*4)
    core=0
    for t in range(steps):
        length=n+(t*s if history else 0)
        core+=2*s*2*d*d+4*s*length*d+2*s*4*d*d+4*s*s*d+2*s*2*d*512
        if history and t+1<steps: core+=2*s*2*d*d
    return encoder+head+core
