"""MLPs without search rules. The readout never feeds back into the core."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Linear(256,256),nn.Sigmoid())

    def forward(self,problem,steps=8):
        if problem.ndim!=2 or problem.shape[1]!=768 or steps<1:
            raise ValueError('Expected problem (batch,768) and positive repeat length')
        valid=problem[:,:256]
        state=problem[:,512:768]
        states=[]
        for _ in range(steps):
            state=self.net(torch.cat((problem,state),dim=1))*valid
            states.append(state)
        return states


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2816,256),nn.ReLU(),nn.Linear(256,256))

    def forward(self,features):
        if features.ndim!=2 or features.shape[1]!=2816:
            raise ValueError('Expected readout features (batch,2816)')
        return self.net(features)


class Baseline(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(768,width),nn.ReLU(),nn.Linear(width,256))

    def forward(self,problem):
        return self.net(problem)


def features(problem,history,mode):
    """No labels are accepted here. History must be model-generated [N,8,256]."""
    if mode=='E':
        record=problem.new_zeros((len(problem),2048))
    elif mode=='C':
        record=history[:,-1:].expand(-1,8,-1).reshape(len(problem),2048)
    elif mode=='D':
        record=history.reshape(len(problem),2048)
    else:
        raise ValueError('Expected C, D or E')
    return torch.cat((problem,record),dim=1)


def masked_loss(prediction,target,valid,logits=False):
    raw=F.binary_cross_entropy_with_logits(prediction,target,reduction='none') if logits else F.binary_cross_entropy(prediction,target,reduction='none')
    return ((raw*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()


def state_hash(model):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def linear_flops(model):
    return sum(2*m.in_features*m.out_features for m in model.modules() if isinstance(m,nn.Linear))
