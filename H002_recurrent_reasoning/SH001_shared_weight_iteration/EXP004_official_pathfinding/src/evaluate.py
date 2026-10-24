"""Shared strictly descending greedy decoder: no BFS/A*/Dijkstra and no repairs."""
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from model import features,flops

VARIANTS=['zero_all','repeat_last','reverse','shuffle','other_sample_all']+[f'zero_R{k}' for k in range(1,9)]


def decode(scores,walls,start,goal,move_limit=1024,epsilon=0.):
    scores=np.asarray(scores).reshape(1024)
    blocked=np.asarray(walls).reshape(1024).astype(bool)
    path=[int(start)]; visited={int(start)}; illegal_best=0; decisions=0
    if not np.isfinite(scores).all():
        return {'path':path,'status':'nonfinite','illegal_best':0,'decisions':0}
    for _ in range(move_limit):
        current=path[-1]
        if current==goal:
            return {'path':path,'status':'success','illegal_best':illegal_best,'decisions':decisions}
        y,x=divmod(current,32)
        neighbors=[yy*32+xx for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if 0<=yy<32 and 0<=xx<32]
        decisions+=1
        raw=min(neighbors,key=lambda i:float(scores[i]))
        illegal_best+=int(blocked[raw])
        candidates=[i for i in neighbors if not blocked[i] and scores[i]<scores[current]-epsilon]
        if not candidates:
            return {'path':path,'status':'local_minimum','illegal_best':illegal_best,'decisions':decisions}
        nxt=min(candidates,key=lambda i:float(scores[i]))
        if nxt in visited:
            return {'path':path,'status':'loop','illegal_best':illegal_best,'decisions':decisions}
        path.append(nxt); visited.add(nxt)
    return {'path':path,'status':'success' if path[-1]==goal else 'move_limit','illegal_best':illegal_best,'decisions':decisions}


def legal_path(path,walls,start,goal=None):
    if not path or path[0]!=start:
        return False
    flat=np.asarray(walls).reshape(1024)
    for i,node in enumerate(path):
        if not 0<=node<1024 or flat[node]:
            return False
        if i:
            py,px=divmod(path[i-1],32); y,x=divmod(node,32)
            if abs(py-y)+abs(px-x)!=1:
                return False
    return goal is None or path[-1]==goal


def evaluate_predictions(predictions,data,indices,config):
    values={k:[] for k in ('success','legal','optimal','spl','cost_ratio','local_minimum','loop','move_limit',
                           'distance_mae','distance_rmse','start_mae','goal_score','illegal_preference','illegal_best','decisions','path_length')}
    paths=[]
    for prediction,index in zip(predictions,indices):
        walls=data['walls'][data['map_id'][index]]
        start,goal=int(data['start'][index]),int(data['goal'][index])
        target=data['fields'][data['field_id'][index]]
        result=decode(prediction,walls,start,goal,config['decoder_move_limit'],config['decoder_strict_descent_epsilon'])
        path=result['path']; success=result['status']=='success'; length=len(path)-1; optimal=int(data['optimal'][index])
        legal=legal_path(path,walls,start)
        if success:
            assert legal_path(path,walls,start,goal) and length>=optimal
        reachable=target>=0
        errors=prediction[reachable].astype(np.float64)*config['distance_scale']-target[reachable]
        record={'success':float(success),'legal':float(legal),'optimal':float(success and length==optimal),
                'spl':float(success*optimal/max(length,optimal)),'cost_ratio':length/optimal if success else float('nan'),
                'local_minimum':float(result['status']=='local_minimum'),'loop':float(result['status']=='loop'),
                'move_limit':float(result['status']=='move_limit'),'distance_mae':float(np.abs(errors).mean()),
                'distance_rmse':float(np.sqrt(np.square(errors).mean())),
                'start_mae':abs(float(prediction[start])*config['distance_scale']-optimal),
                'goal_score':float(prediction[goal])*config['distance_scale'],
                'illegal_preference':result['illegal_best']/max(result['decisions'],1),
                'illegal_best':float(result['illegal_best']),'decisions':float(result['decisions']),
                'path_length':float(length)}
        for key,value in record.items():
            values[key].append(value)
        paths.append({'record_index':int(index),'path':path,'status':result['status'],
                      'optimal_four_way':optimal,'illegal_best':result['illegal_best'],'decisions':result['decisions']})
    return {k:np.array(v) for k,v in values.items()},paths


def perturb(history,variant):
    if variant=='zero_all': return torch.zeros_like(history)
    if variant=='repeat_last': return history[:,-1:].expand(-1,8,-1)
    if variant=='reverse': return history.flip(1)
    if variant=='shuffle': return history[:,[3,0,6,1,7,4,2,5]]
    if variant=='other_sample_all': return history.roll(1,0)
    if variant.startswith('zero_R'):
        changed=history.clone(); changed[:,int(variant[6:])-1]=0
        return changed
    if variant=='normal': return history
    raise ValueError(variant)


@torch.no_grad()
def histories(core,problem,batch=256):
    core.eval()
    return torch.cat([torch.stack(core(p,8),1) for p in problem.split(batch)]).detach()


@torch.no_grad()
def predict(model,problem,history,mode,batch=256,variant='normal'):
    if mode=='B': return history[:,-1].cpu().numpy()
    model.eval()
    history=perturb(history,variant) if mode=='D' else history
    outputs=[]
    for begin in range(0,len(problem),batch):
        p=problem[begin:begin+batch]
        h=history[begin:begin+batch] if history is not None else None
        out=model(features(p,h,mode)) if mode in ('C','D','E') else model(p)
        if not torch.isfinite(out).all(): raise FloatingPointError('Nonfinite prediction')
        outputs.append(out)
    return torch.cat(outputs).cpu().numpy()


def grouping(data,indices):
    groups={}
    for split_id,name in [(2,'seen'),(3,'unseen')]:
        mask=data['split'][indices]==split_id
        groups[name]=mask
        for k in range(4): groups[f'{name}/Q{k+1}']=mask&(data['band'][indices]==k)
        for map_id in range(5): groups[f'{name}/map{map_id}']=mask&(data['map_id'][indices]==map_id)
    return groups


def summarize(values,groups,mode,seed,variant='normal'):
    rows=[]
    for group,mask in groups.items():
        row={'mode':mode,'seed':seed,'variant':variant,'group':group,'n':int(mask.sum())}
        if mask.any():
            row.update({key:float(value[mask].mean()) for key,value in values.items() if key!='cost_ratio'})
            success=mask&(values['success']>0)
            row['success_cost_ratio']=float(values['cost_ratio'][success].mean()) if success.any() else None
            row['optimal_fraction_among_success']=float(values['optimal'][success].mean()) if success.any() else None
            row['illegal_best_event_rate']=float(values['illegal_best'][mask].sum()/max(values['decisions'][mask].sum(),1))
        rows.append(row)
    return rows


def aggregate(rows):
    result=[]
    for mode,variant,group in sorted({(r['mode'],r['variant'],r['group']) for r in rows}):
        selected=[r for r in rows if (r['mode'],r['variant'],r['group'])==(mode,variant,group)]
        row={'mode':mode,'variant':variant,'group':group,'n':selected[0]['n'],'seeds':len(selected)}
        if row['n']:
            for key in selected[0]:
                if key in ('mode','seed','variant','group','n'): continue
                vals=[r[key] for r in selected if r[key] is not None]
                row[key+'_mean']=float(np.mean(vals)) if vals else None
                row[key+'_std']=float(np.std(vals,ddof=1 if len(vals)>1 else 0)) if vals else None
        result.append(row)
    return result


@torch.no_grad()
def latency(model,core,host,device,mode,config):
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    core.eval()
    if model is not None: model.eval()
    def forward(p):
        if mode=='B': return core(p,8)[-1]
        if mode in ('C','D'): return model(features(p,torch.stack(core(p,8),1),mode))
        if mode=='E': return model(features(p,None,'E'))
        return model(p)
    p=host.to(device)
    for _ in range(config['latency_warmup']): forward(p)
    sync()
    base=torch.cuda.memory_allocated() if device.type=='cuda' else 0
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    info={'batch_size':len(p),'warmup':config['latency_warmup'],'repeats':config['latency_repeats']}
    for kind in ('model_only','end_to_end','model_and_decoder'):
        times=[]
        for _ in range(config['latency_repeats']):
            sync(); started=time.perf_counter()
            out=forward(host.to(device)).cpu() if kind!='model_only' else forward(p)
            if kind=='model_and_decoder':
                for score,original in zip(out.numpy(),host.numpy()):
                    decode(score,original[:1024],int(original[1024:2048].argmax()),
                           int(original[2048:].argmax()),config['decoder_move_limit'],config['decoder_strict_descent_epsilon'])
            sync(); times.append((time.perf_counter()-started)*1000)
        info[kind+'_ms_mean']=float(np.mean(times)); info[kind+'_ms_std']=float(np.std(times))
    info['inference_peak_incremental_bytes']=torch.cuda.max_memory_allocated()-base if device.type=='cuda' else None
    info['core_flops']=8*flops(core) if mode in ('B','C','D') else 0
    info['head_flops']=flops(model) if mode in ('C','D','E') else 0
    info['total_flops']=info['core_flops']+info['head_flops'] if mode in ('B','C','D','E') else flops(model)
    info['history_bytes_per_sample']=8*1024*4 if mode in ('C','D') else 0
    return info
