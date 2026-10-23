"""Valid-cell metrics, frozen-core caches, history interventions and real system cost."""
import time
import numpy as np
import torch
from model import features,linear_flops


def sample_metrics(probabilities,problem,distances,start,logits=None):
    valid=problem[:,:256].astype(bool)
    truth=distances>=0
    pred=(probabilities>=.5)&valid
    nvalid=valid.sum(1)
    intersection=(pred&truth).sum(1)
    union=(pred|truth).sum(1)
    fden=pred.sum(1)+truth.sum(1)
    p=probabilities.astype(np.float64)
    if logits is None:
        with np.errstate(divide='ignore'):
            loss=-(truth*np.maximum(np.log(p),-100)+(~truth)*np.maximum(np.log1p(-p),-100))
    else:
        z=logits.astype(np.float64)
        loss=np.logaddexp(0,z)-truth*z
    unreachable=valid&(problem[:,256:512]==0)&~truth
    wrong=(pred!=truth)&valid
    confidence=(p>=.9)|(p<=.1)
    return {'iou':intersection/np.maximum(union,1),
            'f1':2*intersection/np.maximum(fden,1),
            'bce':(loss*valid).sum(1)/nvalid,
            'exact':((pred==truth)|~valid).all(1).astype(float),
            'start_accuracy':(pred[np.arange(len(pred)),start]==truth[np.arange(len(pred)),start]).astype(float),
            'fnr':((~pred)&truth).sum(1)/np.maximum(truth.sum(1),1),
            'disconnected_fpr':(pred&unreachable).sum(1)/np.maximum(unreachable.sum(1),1),
            'confident_wrong':(wrong&confidence).sum(1)/nvalid,
            'pixel_error':wrong.sum(1)/nvalid}


@torch.no_grad()
def cache_history(core,problems,batch_size=256):
    core.eval()
    chunks=[]
    for problem in problems.split(batch_size):
        chunks.append(torch.stack(core(problem,8),dim=1))
    return torch.cat(chunks).detach()


def intervention(history,name):
    if name=='normal':
        return history
    if name=='repeat_last':
        return history[:,-1:].expand(-1,8,-1)
    if name=='zero_all':
        return torch.zeros_like(history)
    if name=='reverse':
        return history.flip(1)
    if name=='shuffle':
        return history[:,[3,0,6,1,7,4,2,5]]
    if name=='other_sample_all':
        return history.roll(1,0)
    altered=history.clone()
    if name.startswith('zero_R'):
        altered[:,int(name[6:])-1]=0
    elif name=='other_sample_R4':
        altered[:,3]=history.roll(1,0)[:,3]
    else:
        raise ValueError(name)
    return altered


INTERVENTIONS=['repeat_last','zero_all','reverse','shuffle','other_sample_all','other_sample_R4']+[f'zero_R{k}' for k in range(1,9)]


@torch.no_grad()
def predict(model,problems,history=None,mode='A',batch_size=256,perturbation='normal'):
    if mode=='B':
        return history[:,-1].cpu().numpy(),None
    model.eval()
    changed=intervention(history,perturbation) if history is not None and mode=='D' else history
    logits=[]
    for begin in range(0,len(problems),batch_size):
        p=problems[begin:begin+batch_size]
        if mode in ('C','D','E'):
            h=changed[begin:begin+batch_size] if changed is not None else None
            out=model(features(p,h,mode))
        else:
            out=model(p)
        if not torch.isfinite(out).all():
            raise FloatingPointError('Nonfinite prediction logits')
        logits.append(out)
    logits=torch.cat(logits)
    return logits.sigmoid().cpu().numpy(),logits.cpu().numpy()


@torch.no_grad()
def latency(model,core,host_problem,device,mode,config):
    """Measures the actual system; C/D recompute 8 states, E skips the core."""
    def sync():
        if device.type=='cuda':
            torch.cuda.synchronize()
    if model is not None:
        model.eval()
    core.eval()
    def forward(problem):
        if mode=='B':
            return core(problem,8)[-1]
        if mode in ('C','D'):
            history=torch.stack(core(problem,8),dim=1)
            return model(features(problem,history,mode)).sigmoid()
        if mode=='E':
            return model(features(problem,None,'E')).sigmoid()
        return model(problem).sigmoid()
    problem=host_problem.to(device)
    for _ in range(config['latency_warmup']):
        forward(problem)
    sync()
    baseline=torch.cuda.memory_allocated() if device.type=='cuda' else None
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    result={'batch_size':len(problem),'warmup':config['latency_warmup'],'repeats':config['latency_repeats'],
            'core_calls_per_batch':8 if mode in ('B','C','D') else 0,
            'head_calls_per_batch':1 if mode in ('C','D','E') else 0,
            'history_bytes_per_batch':len(problem)*8*256*4 if mode in ('C','D') else 0}
    for kind in ('model_only','end_to_end'):
        times=[]
        for _ in range(config['latency_repeats']):
            sync()
            started=time.perf_counter()
            out=forward(host_problem.to(device)).cpu() if kind=='end_to_end' else forward(problem)
            sync()
            times.append((time.perf_counter()-started)*1000)
        result[kind+'_ms_mean']=float(np.mean(times))
        result[kind+'_ms_std']=float(np.std(times))
    result['inference_peak_incremental_allocated_bytes']=torch.cuda.max_memory_allocated()-baseline if device.type=='cuda' else None
    result['core_linear_flops_per_sample']=8*linear_flops(core) if mode in ('B','C','D') else 0
    result['head_linear_flops_per_sample']=linear_flops(model) if mode in ('C','D','E') else 0
    result['total_linear_flops_per_sample']=result['core_linear_flops_per_sample']+result['head_linear_flops_per_sample'] if mode in ('B','C','D','E') else linear_flops(model)
    return result
