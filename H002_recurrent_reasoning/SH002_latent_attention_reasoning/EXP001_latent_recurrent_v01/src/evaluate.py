"""Unmasked argmax policy routing and actual fixed/adaptive inference."""
import hashlib
import time
import numpy as np
from labels import MOVES,policy_loss
from common import write


def decode(logits,walls,start,goal,limit=1024):
    path=[int(start)]; seen={int(start)}; at=int(start)
    for _ in range(limit):
        if at==goal: return {'status':'success','path':path}
        row=logits[at]
        if not np.isfinite(row).all(): return {'status':'nonfinite','path':path}
        action=int(np.argmax(row)); y,x=divmod(at,32); dy,dx=MOVES[action]; yy,xx=y+dy,x+dx
        if not (0<=yy<32 and 0<=xx<32): return {'status':'out_of_bounds','path':path,'attempt':action}
        nxt=yy*32+xx
        if walls[nxt]: return {'status':'wall','path':path,'attempt':action}
        path.append(nxt)
        if nxt in seen: return {'status':'revisit','path':path}
        seen.add(nxt); at=nxt
    return {'status':'success' if at==goal else 'move_limit','path':path}


def values_for(logits,correct,data,ids,times=None,forced=None):
    import torch
    losses=policy_loss(torch.from_numpy(logits),torch.from_numpy(correct)).numpy()
    p=torch.from_numpy(logits).softmax(-1).numpy()
    metrics={k:[] for k in ('success','optimal','spl','cost_ratio','wall','out_of_bounds','revisit','move_limit','nonfinite','loss','steps','forced','high_confidence_wrong')}
    paths=[]
    for j,i in enumerate(ids):
        route=decode(logits[j],data['walls'][data['map_id'][i]],int(data['start'][i]),int(data['goal'][i]))
        length=len(route['path'])-1; ok=route['status']=='success'; optimum=int(data['optimal'][i])
        action=logits[j].argmax(-1); valid=correct[j].any(-1)
        wrong=~correct[j,np.arange(1024),action]
        confidence=p[j].max(-1)
        row={'success':float(ok),'optimal':float(ok and length==optimum),'spl':float(ok)*optimum/max(length,optimum),
             'cost_ratio':length/optimum if ok else np.nan,'loss':float(losses[j]),
             'steps':int(times[j]) if times is not None else 0,'forced':float(forced[j]) if forced is not None else 0.,
             'high_confidence_wrong':float((wrong&valid&(confidence>=.9)).sum()/max(valid.sum(),1)),
             **{k:float(route['status']==k) for k in ('wall','out_of_bounds','revisit','move_limit','nonfinite')}}
        for k,v in row.items(): metrics[k].append(v)
        paths.append({'record_index':int(i),**route,'length':length,'optimal_length':optimum,'steps':row['steps'],'forced':bool(row['forced'])})
    return {k:np.asarray(v) for k,v in metrics.items()},paths


def groups(data,ids):
    result={}
    for split,name in ((1,'val'),(2,'seen'),(3,'unseen')):
        base=data['split'][ids]==split
        if not base.any(): continue
        result[name]=base
        for k in range(4): result[f'{name}/Q{k+1}']=base&(data['band'][ids]==k)
        for k in range(5): result[f'{name}/map{k}']=base&(data['map_id'][ids]==k)
    return result


def summarize(metrics,data,ids):
    output={}
    for name,mask in groups(data,ids).items():
        row={'n':int(mask.sum())}
        if mask.any():
            row.update({k:float(v[mask].mean()) for k,v in metrics.items() if k!='cost_ratio'})
            success=mask&(metrics['success']>0)
            row['success_cost_ratio']=float(metrics['cost_ratio'][success].mean()) if success.any() else None
            row['optimal_among_success']=float(metrics['optimal'][success].mean()) if success.any() else None
            row['early_success']=float(metrics['success'][mask&(metrics['steps']<16)].mean()) if (mask&(metrics['steps']<16)).any() else None
            row['first_stop_fraction']=float((metrics['steps'][mask]==1).mean())
            row['step_histogram']=[int((metrics['steps'][mask]==t).sum()) for t in range(1,17)]
        output[name]=row
    return output


def evaluate(model,dataset,ids,steps=16,history=True,batch=64,halt=None,variant='normal',save=None,expected_hash=None):
    import torch
    model.eval()
    output=[]; times=[]; forced=[]
    with torch.no_grad():
        for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/batch)))):
            tokens,_=dataset.batch(part)
            if halt is None:
                logits=model(tokens,steps,history,variant); times.extend([steps]*len(part)); forced.extend([False]*len(part))
            else:
                logits,info=model.adaptive(tokens,halt); times.extend(info['steps'].cpu().tolist()); forced.extend(info['forced'].cpu().tolist())
            output.append(logits.cpu().numpy())
    output=np.concatenate(output)
    if expected_hash is not None: assert hashlib.sha256(output.tobytes()).hexdigest()==expected_hash
    correct=dataset.labels[dataset.data['field_id'][ids]]
    metrics,paths=values_for(output,correct,dataset.data,ids,np.array(times),np.array(forced))
    rows=summarize(metrics,dataset.data,ids)
    if save:
        np.save(str(save)+'_logits.npy',output)
        np.savez_compressed(str(save)+'_metrics.npz',**metrics)
        write(str(save)+'_paths.json',paths); write(str(save)+'_groups.json',rows)
        write(str(save)+'_prediction_hash.json',{'sha256':hashlib.sha256(output.tobytes()).hexdigest(),'shape':list(output.shape)})
    return rows,metrics,paths


def latency(model,dataset,ids,steps,history,halt=None,repeats=20):
    import torch
    from model import estimated_flops
    records=[]
    with torch.no_grad():
        for batch in (1,16,64):
            x,_=dataset.batch(ids[:batch]); x=x.clone()
            def forward():
                return model(x,steps,history) if halt is None else model.adaptive(x,halt)[0]
            for _ in range(5): forward()
            torch.cuda.synchronize(); base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            values=[]
            for _ in range(repeats):
                torch.cuda.synchronize(); start=time.perf_counter(); forward(); torch.cuda.synchronize()
                values.append(1000*(time.perf_counter()-start))
            records.append({'batch':batch,'mean_ms':float(np.mean(values)),'max_ms':max(values),'std_ms':float(np.std(values)),
                            'peak_increment_bytes':torch.cuda.max_memory_allocated()-base,
                            'fixed_forward_flops_per_sample':estimated_flops(steps,history) if halt is None else None})
    return records
