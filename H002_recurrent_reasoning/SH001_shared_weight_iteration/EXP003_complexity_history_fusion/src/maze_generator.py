"""Variable-size mazes embedded in a 16x16 canvas; BFS only produces labels."""
from collections import deque
import hashlib
import numpy as np

SPLITS=('train','val','test')
META=['size','eccentricity','branches','dead_ends','components','unreachable_open_fraction',
      'wall_density','mean_detour_ratio','max_detour_ratio','family','start_distance','cycle_rank']


def neighbors(i,n):
    r,c=divmod(int(i),n)
    return [rr*n+cc for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if 0<=rr<n and 0<=cc<n]


def bfs(walls,goal,n):
    walls=np.asarray(walls).reshape(-1)
    d=np.full(n*n,-1,dtype=np.int16)
    if walls[goal]:
        raise ValueError('Blocked goal')
    d[goal]=0
    queue=deque([int(goal)])
    while queue:
        i=queue.popleft()
        for j in neighbors(i,n):
            if not walls[j] and d[j]<0:
                d[j]=d[i]+1
                queue.append(j)
    return d


def canonical(walls):
    grid=walls.astype(np.uint8)
    return min(np.packbits(x).tobytes() for k in range(4) for x in (np.rot90(grid,k),np.rot90(grid[:,::-1],k)))


def candidate(n,rng,family):
    if family==0:
        return (rng.random((n,n))<rng.uniform(.25,.50)).astype(np.uint8)
    walls=np.ones((n,n),dtype=np.uint8)
    coarse=n//2
    first=(int(rng.integers(coarse)),int(rng.integers(coarse)))
    seen,stack,passages={first},[first],[]
    walls[first[0]*2,first[1]*2]=0
    while stack:
        r,c=stack[-1]
        choices=[(rr,cc) for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                 if 0<=rr<coarse and 0<=cc<coarse and (rr,cc) not in seen]
        if not choices:
            stack.pop()
            continue
        rr,cc=choices[int(rng.integers(len(choices)))]
        walls[r+rr,c+cc]=walls[2*rr,2*cc]=0
        passages.append((r+rr,c+cc))
        seen.add((rr,cc))
        stack.append((rr,cc))
    # Cut random tree edges, creating irregular disconnected components, no full barrier.
    cut_count=int(rng.integers(1,max(3,n//2)))
    for index in rng.choice(len(passages),cut_count,replace=False):
        walls[passages[index]]=1
    return walls


def topology(walls,goal,d,n):
    opened=np.flatnonzero(walls.reshape(-1)==0)
    degrees=[sum(not walls.reshape(-1)[j] for j in neighbors(i,n)) for i in opened]
    remaining=set(map(int,opened))
    components=0
    while remaining:
        component=bfs(walls,min(remaining),n)>=0
        remaining.difference_update(np.flatnonzero(component).tolist())
        components+=1
    locations=np.flatnonzero(d>0)
    gr,gc=divmod(goal,n)
    manhattan=np.abs(locations//n-gr)+np.abs(locations%n-gc)
    detours=d[locations]/np.maximum(manhattan,1)
    return [int((np.array(degrees)>=3).sum()),int((np.array(degrees)==1).sum()),components,
            float(((d<0)&(walls.reshape(-1)==0)).sum()/len(opened)),float(walls.mean()),
            float(detours.mean()),float(detours.max()),int(sum(degrees)//2-len(opened)+components)]


def difficulty(eccentricity):
    return 'Easy' if eccentricity<=8 else 'Medium' if eccentricity<=16 else 'Hard' if eccentricity<=32 else 'Extreme'


def generate(config):
    rng=np.random.default_rng(config['data_seed'])
    blocks={s:[] for s in SPLITS}
    generation={'seed':config['data_seed'],'metadata_columns':META,'sizes':{}}
    count=sum(config['sizes_per_maze_size'])
    for n in config['maze_sizes']:
        seen,records,attempts=set(),[],0
        while len(records)<count:
            attempts+=1
            if attempts>count*500:
                raise RuntimeError(f'Generation exhausted size={n}; accepted={len(records)}; no criteria relaxed')
            family=len(records)%2
            walls=candidate(n,rng,family)
            key=canonical(walls)
            if key in seen:
                continue
            opened=np.flatnonzero(walls.reshape(-1)==0)
            if not len(opened):
                continue
            goal=int(rng.choice(opened))
            dist=bfs(walls,goal,n)
            disconnected=(dist<0)&(walls.reshape(-1)==0)
            if dist.max()<config['minimum_eccentricity'] or disconnected.sum()<2 or disconnected.sum()/len(opened)<config['minimum_unreachable_open_fraction']:
                continue
            t=topology(walls,goal,dist,n)
            records.append((walls,goal,dist,family,t))
            seen.add(key)
        order=rng.permutation(count)
        offset=0
        for split,size in zip(SPLITS,config['sizes_per_maze_size']):
            balanced=np.arange(size)%2==0
            rng.shuffle(balanced)
            for k,index in enumerate(order[offset:offset+size]):
                walls,goal,dist,family,t=records[index]
                starts=np.flatnonzero(dist>0) if balanced[k] else np.flatnonzero((dist<0)&(walls.reshape(-1)==0))
                start=int(rng.choice(starts))
                rotation=int(rng.integers(4))
                flip=bool(rng.integers(2))
                # Transform W/G/S/dist together; canvas offset varies independently.
                local_g=np.zeros((n,n),dtype=np.float32)
                local_g.flat[goal]=1
                local_s=np.zeros((n,n),dtype=np.uint8)
                local_s.flat[start]=1
                def transform(a):
                    a=np.rot90(a,rotation)
                    return a[:,::-1] if flip else a
                off_r,off_c=[int(rng.integers(17-n)) for _ in range(2)]
                valid=np.zeros((16,16),dtype=np.float32)
                valid[off_r:off_r+n,off_c:off_c+n]=1
                def embed(a,fill=0):
                    canvas=np.full((16,16),fill,dtype=a.dtype)
                    canvas[off_r:off_r+n,off_c:off_c+n]=transform(a)
                    return canvas.reshape(256)
                problem=np.concatenate((valid.reshape(256),embed(walls.astype(np.float32)),embed(local_g)))
                global_s=int(embed(local_s).argmax())
                meta=np.array([n,int(dist.max()),t[0],t[1],t[2],t[3],t[4],t[5],t[6],family,int(dist[start]),t[7]],dtype=np.float64)
                blocks[split].append((problem,embed(dist.reshape(n,n),-1),global_s,meta))
            offset+=size
        generation['sizes'][str(n)]={'attempts':attempts,'accepted':len(records),'unique_up_to_rotation_reflection':len(seen)}
    data={}
    for split,records in blocks.items():
        order=rng.permutation(len(records))
        records=[records[i] for i in order]
        for key,array in {'problem':np.array([r[0] for r in records],dtype=np.float32),
                          'distances':np.array([r[1] for r in records],dtype=np.int16),
                          'start':np.array([r[2] for r in records],dtype=np.int64),
                          'meta':np.array([r[3] for r in records],dtype=np.float64)}.items():
            data[split+'_'+key]=array
    return data,generation


def local_map(problem):
    valid=problem[:256].reshape(16,16).astype(bool)
    rr,cc=np.where(valid)
    n=int(round(np.sqrt(valid.sum())))
    sl=(slice(rr.min(),rr.min()+n),slice(cc.min(),cc.min()+n))
    return problem[256:512].reshape(16,16)[sl],problem[512:].reshape(16,16)[sl],sl,n


def describe(x):
    x=np.asarray(x)
    return {'min':float(x.min()),'mean':float(x.mean()),'max':float(x.max()),
            'percentiles_25_50_75_90':np.percentile(x,[25,50,75,90]).tolist()}


def audit(data,config):
    report,sets={},{}
    for split in SPLITS:
        p,d,meta,start=[data[split+'_'+k] for k in ('problem','distances','meta','start')]
        assert p.shape==(len(p),768) and d.shape==(len(p),256)
        keys=set()
        for problem,dist in zip(p,d):
            walls,goal,sl,n=local_map(problem)
            key=(n,canonical(walls))
            assert key not in keys
            keys.add(key)
            valid=problem[:256].astype(bool)
            assert valid.sum()==n*n and problem[512:].sum()==1
            assert np.all(problem[256:512][~valid]==0) and np.all(dist[~valid]==-1)
            assert np.array_equal(bfs(walls,int(goal.argmax()),n),dist.reshape(16,16)[sl].reshape(-1))
        sets[split]=keys
        cells=p[:,:256].sum(1)
        final=d>=0
        free=(p[:,256:512]==0)&p[:,:256].astype(bool)
        report[split]={'n':len(p),'sha256':hashlib.sha256(b''.join(data[split+'_'+k].tobytes() for k in ('problem','distances','start','meta'))).hexdigest(),
                       'all_open_iou':float((final.sum(1)/free.sum(1)).mean()),
                       'all_zero_valid_pixel_error':float((final.sum(1)/cells).mean()),
                       'metadata':{key:describe(meta[:,i]) for i,key in enumerate(META)},'size_difficulty':{}}
        for n in config['maze_sizes']:
            m=meta[:,0]==n
            assert m.sum()==config['sizes_per_maze_size'][SPLITS.index(split)]
            start_actual=final[np.arange(len(p)),start]
            assert start_actual[m].sum()==(int(m.sum())+1)//2
            for band,(lo,hi) in config['difficulty_bands'].items():
                mask=m&(meta[:,1]>=lo)&(meta[:,1]<=hi)
                report[split]['size_difficulty'][f'{n}/{band}']={'n':int(mask.sum()),
                    'status':'observed' if mask.any() else 'not_generated; no relaxation or imputation'}
        assert np.all(((free & ~final).sum(1)/free.sum(1))>=config['minimum_unreachable_open_fraction'])
    assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    report['split_overlap_up_to_rotations_reflections']=0
    return report
