"""Measure the unmodified FP32 BPTT graph before selecting batch size."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import gc
import time
import numpy as np
import torch
from common import BASE,setup,seed_all,write,environment
from dataset import Dataset
from model import Solver,parameters,estimated_flops
from labels import policy_loss


def main():
    device=setup(); seed_all(42); ds=Dataset(device); result=[]
    for batch in (16,32,64,128,256,512,768):
        model=Solver().to(device); opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        x,y=ds.batch(ds.ids['train'][:batch]); torch.cuda.reset_peak_memory_stats()
        def step():
            opt.zero_grad(set_to_none=True); logits=model(x,16,True)
            value=policy_loss(logits,y).mean(); value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        try:
            step(); torch.cuda.synchronize(); times=[]
            for _ in range(3):
                start=time.perf_counter(); step(); torch.cuda.synchronize(); times.append(time.perf_counter()-start)
            row={'batch':batch,'seconds_per_train_batch':float(np.mean(times)),'samples_per_second':batch/float(np.mean(times)),
                 'peak_allocated':torch.cuda.max_memory_allocated(),'parameters':parameters(model),
                 'estimated_forward_flops':estimated_flops(16,True),'success':True}
        except torch.cuda.OutOfMemoryError as exc:
            row={'batch':batch,'success':False,'OOM':str(exc)}
        result.append(row); print(row,flush=True)
        write(BASE/'results/resource_benchmark.json',{'environment':environment(),'measurements':result})
        del model,opt,x,y; gc.collect(); torch.cuda.empty_cache()
        if not row['success']: break
    eligible=[r for r in result if r['success'] and r['peak_allocated']<24*2**30]
    best=max(eligible,key=lambda r:r['samples_per_second'])
    write(BASE/'results/resource_benchmark.json',{'environment':environment(),'measurements':result,'recommended_batch':best['batch'],
        'rule':'highest measured throughput among candidate batches with <24 GiB allocated; FP32 final-only 16-step BPTT unchanged'})


if __name__=='__main__': main()
