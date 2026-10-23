"""Frozen-core readouts and single-pass baselines share the same training loop."""
import copy
import hashlib
import json
import time
import numpy as np
import torch
from model import Readout,Baseline,features,masked_loss,state_hash,parameters,linear_flops


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def fit(run,checkpoints,config,device,tensors,histories,core,seed,mode,initial=None):
    torch.manual_seed(seed+50000)
    model=Readout() if mode in ('C','D','E') else Baseline(config['baseline_widths'][mode])
    if initial is not None:
        model.load_state_dict(initial)
    initial_hash=state_hash(model)
    model=model.to(device)
    core_hash=state_hash(core)
    assert all(not p.requires_grad and p.grad is None for p in core.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(seed+10000)
    order_hash=hashlib.sha256()
    name=f'{mode}_seed{seed}'
    epochs=config['epochs_heads'] if mode in ('C','D','E') else config['epochs_baselines']
    history,steps,max_grad=[],0,0.
    device_baseline=torch.cuda.memory_allocated() if device.type=='cuda' else None
    if device.type=='cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    def inputs(split,indices):
        problem=tensors[split]['problem'][indices]
        if mode in ('C','D','E'):
            h=histories[split][indices] if mode!='E' else None
            return features(problem,h,mode)
        return problem
    try:
        for epoch in range(epochs):
            model.train()
            host_order=torch.randperm(len(tensors['train']['problem']),generator=generator)
            order_hash.update(host_order.numpy().tobytes())
            losses,epoch_grad=[],0.
            for indices in host_order.to(device).split(config['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                logits=model(inputs('train',indices))
                valid=tensors['train']['problem'][indices,:256]
                target=(tensors['train']['distances'][indices]>=0).float()
                loss=masked_loss(logits,target,valid,logits=True)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite logits/loss')
                loss.backward()
                norm=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
                if not np.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient')
                epoch_grad=max(epoch_grad,norm)
                optimizer.step()
                steps+=1
                losses.append(loss.item())
            max_grad=max(max_grad,epoch_grad)
            model.eval()
            with torch.no_grad():
                losses_val=[]
                for indices in torch.arange(len(tensors['val']['problem']),device=device).split(256):
                    z=model(inputs('val',indices))
                    target=(tensors['val']['distances'][indices]>=0).float()
                    losses_val.append(masked_loss(z,target,tensors['val']['problem'][indices,:256],logits=True).item()*len(indices))
                validation=sum(losses_val)/len(tensors['val']['problem'])
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),
                            'validation_final_bce':validation,'gradient_norm_max':epoch_grad})
            if epoch==0 or (epoch+1)%20==0:
                print(f'{name} epoch={epoch+1} train={np.mean(losses):.5f} val={validation:.5f}',flush=True)
    except Exception:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoints/(name+'_FAILED.pt'))
        write(run/(name+'_history_FAILED.json'),history)
        raise
    if device.type=='cuda':
        torch.cuda.synchronize()
    duration=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
    assert core_hash==state_hash(core)
    assert all(not p.requires_grad and p.grad is None for p in core.parameters())
    record={'name':name,'mode':mode,'seed':seed,'epochs':epochs,'initial_sha256':initial_hash,
            'batch_order_sha256':order_hash.hexdigest(),'core_sha256':core_hash,
            'optimizer_steps':steps,'own_train_seconds':duration,'train_peak_allocated_bytes':peak,
            'train_peak_incremental_bytes':peak-device_baseline if peak is not None else None,
            'gradient_norm_max':max_grad,'trainable_parameters':parameters(model),
            'loss_connected_parameters':parameters(Baseline(256)) if mode=='E' else parameters(model),
            'train_head_calls':steps if mode in ('C','D','E') else 0,
            'train_baseline_calls':steps if mode not in ('C','D','E') else 0,
            'own_train_forward_linear_flops':linear_flops(model)*len(tensors['train']['problem'])*epochs,
            'core_unchanged_after_training':True}
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),**record},checkpoints/(name+'.pt'))
    write(run/(name+'_history.json'),history)
    return model,record


def head_initial(seed):
    torch.manual_seed(seed+50000)
    return copy.deepcopy(Readout().state_dict())
