"""Distance-score MLPs; no shortest-path code or labels in any forward method."""
import hashlib
import torch
from torch import nn


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4096,256),nn.ReLU(),nn.Linear(256,1024),nn.Softplus())

    def forward(self,problem,steps=8):
        if problem.ndim!=2 or problem.shape[1]!=3072:
            raise ValueError('Expected W/S/G, shape (N,3072)')
        state=torch.ones_like(problem[:,:1024])
        states=[]
        for _ in range(steps):
            state=self.net(torch.cat((problem,state),1))
            states.append(state)
        return states


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(11264,256),nn.ReLU(),nn.Linear(256,1024),nn.Softplus())

    def forward(self,x):
        return self.net(x)


class Baseline(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3072,width),nn.ReLU(),nn.Linear(width,1024),nn.Softplus())

    def forward(self,x):
        return self.net(x)


def features(problem,history,mode):
    if mode=='C':
        record=history[:,-1:].expand(-1,8,-1).reshape(len(problem),8192)
    elif mode=='D':
        record=history.reshape(len(problem),8192)
    elif mode=='E':
        record=problem.new_zeros((len(problem),8192))
    else:
        raise ValueError(mode)
    return torch.cat((problem,record),1)


def loss(pred,target,mask):
    return (((pred-target).square()*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()


def state_hash(model):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def flops(model):
    return sum(2*m.in_features*m.out_features for m in model.modules() if isinstance(m,nn.Linear))
