"""Independent graph references and set-valued shortest-direction supervision only."""
from collections import deque
import heapq
import numpy as np

MOVES=((-1,0),(1,0),(0,-1),(0,1))


def bfs(walls,goal):
    w=np.asarray(walls).reshape(1024); d=np.full(1024,-1,dtype=np.int16); d[goal]=0
    queue=deque([int(goal)])
    while queue:
        i=queue.popleft(); y,x=divmod(i,32)
        for dy,dx in MOVES:
            yy,xx=y+dy,x+dx
            if 0<=yy<32 and 0<=xx<32:
                j=yy*32+xx
                if not w[j] and d[j]<0: d[j]=d[i]+1; queue.append(j)
    return d


def dijkstra(walls,goal):
    w=np.asarray(walls).reshape(32,32); d=np.full((32,32),100000,dtype=np.int32)
    gy,gx=divmod(int(goal),32); d[gy,gx]=0; heap=[(0,gy,gx)]
    while heap:
        cost,y,x=heapq.heappop(heap)
        if cost!=d[y,x]: continue
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<32 and 0<=xx<32 and not w[yy,xx] and cost+1<d[yy,xx]:
                d[yy,xx]=cost+1; heapq.heappush(heap,(cost+1,yy,xx))
    return np.where(d==100000,-1,d).astype(np.int16).reshape(1024)


def directions(walls,field):
    w=np.asarray(walls).reshape(32,32); f=field.reshape(32,32)
    label=np.zeros((32,32,4),dtype=bool)
    for y in range(32):
        for x in range(32):
            if w[y,x] or f[y,x]<=0: continue
            for a,(dy,dx) in enumerate(MOVES):
                yy,xx=y+dy,x+dx
                if 0<=yy<32 and 0<=xx<32 and not w[yy,xx] and f[yy,xx]==f[y,x]-1:
                    label[y,x,a]=True
    return label.reshape(1024,4)


def policy_loss(logits,correct):
    import torch
    valid=correct.any(-1)
    # Avoid -inf on unsupervised cells: replace their set with all directions first.
    permitted=correct|(~valid).unsqueeze(-1)
    logp=logits.log_softmax(-1)
    cell=-torch.logsumexp(logp.masked_fill(~permitted,float('-inf')),-1)
    per_problem=(cell*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return per_problem
