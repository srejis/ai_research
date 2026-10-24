"""Label-only BFS and separately implemented heap Dijkstra. Never used by model/decoder."""
from collections import deque
import heapq
import math
import numpy as np


def bfs(walls,goal):
    grid=np.asarray(walls).reshape(32,32)
    dist=np.full((32,32),-1,dtype=np.int16)
    gy,gx=divmod(int(goal),32)
    if grid[gy,gx]:
        raise ValueError('Blocked goal')
    dist[gy,gx]=0
    queue=deque([(gy,gx)])
    while queue:
        y,x=queue.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<32 and 0<=xx<32 and not grid[yy,xx] and dist[yy,xx]<0:
                dist[yy,xx]=dist[y,x]+1
                queue.append((yy,xx))
    return dist.reshape(1024)


def dijkstra(walls,goal,diagonals=False):
    blocked=np.asarray(walls).reshape(-1).astype(bool)
    dist=np.full(1024,np.inf,dtype=np.float64)
    dist[goal]=0.
    heap=[(0.,int(goal))]
    moves=[(-1,0),(1,0),(0,-1),(0,1)]
    if diagonals:
        moves += [(-1,-1),(-1,1),(1,-1),(1,1)]
    while heap:
        cost,index=heapq.heappop(heap)
        if cost!=dist[index]:
            continue
        row,col=divmod(index,32)
        for dr,dc in moves:
            rr,cc=row+dr,col+dc
            if not (0<=rr<32 and 0<=cc<32):
                continue
            nxt=rr*32+cc
            if blocked[nxt]:
                continue
            if dr and dc and (blocked[row*32+cc] or blocked[rr*32+col]):
                continue
            candidate=cost+(math.sqrt(2) if dr and dc else 1.)
            if candidate<dist[nxt]:
                dist[nxt]=candidate
                heapq.heappush(heap,(candidate,nxt))
    if not diagonals:
        return np.where(np.isfinite(dist),dist,-1).astype(np.int16)
    return dist
