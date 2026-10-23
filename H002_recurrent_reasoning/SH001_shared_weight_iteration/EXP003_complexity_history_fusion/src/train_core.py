"""Entrypoint: generate/audit -> step-supervised core -> frozen heads -> baselines -> QA."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
import torch
from maze_generator import generate,audit
from model import Core,Readout,masked_loss,state_hash,parameters,linear_flops
from train_heads import fit as fit_logits,head_initial,write
from evaluate import cache_history,predict,latency,INTERVENTIONS

BASE=Path(__file__).resolve().parents[1]


def fit_core(run,checkpoints,config,device,tensors,seed):
    torch.manual_seed(seed)
    model=Core().to(device)
    initial_hash=state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(seed+10000)
    order_hash=hashlib.sha256()
    name=f'core_seed{seed}'
    history,steps,max_grad=[],0,0.
    before_memory=torch.cuda.memory_allocated() if device.type=='cuda' else None
    if device.type=='cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    try:
        for epoch in range(config['epochs_core']):
            model.train()
            order=torch.randperm(len(tensors['train']['problem']),generator=generator)
            order_hash.update(order.numpy().tobytes())
            losses,epoch_grad=[],0.
            for indices in order.to(device).split(config['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                problem=tensors['train']['problem'][indices]
                dist=tensors['train']['distances'][indices]
                states=model(problem,8)
                loss=torch.stack([masked_loss(r,((dist>=0)&(dist<=k)).float(),problem[:,:256])
                                  for k,r in enumerate(states,1)]).mean()
                if not torch.isfinite(torch.stack(states)).all() or not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite core states/loss')
                loss.backward()
                norm=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
                if not np.isfinite(norm):
                    raise FloatingPointError('Nonfinite core gradient')
                epoch_grad=max(epoch_grad,norm)
                optimizer.step()
                steps+=1
                losses.append(loss.item())
            max_grad=max(max_grad,epoch_grad)
            model.eval()
            val_loss=[]
            with torch.no_grad():
                for indices in torch.arange(len(tensors['val']['problem']),device=device).split(256):
                    problem=tensors['val']['problem'][indices]
                    dist=tensors['val']['distances'][indices]
                    states=model(problem,8)
                    v=torch.stack([masked_loss(r,((dist>=0)&(dist<=k)).float(),problem[:,:256])
                                   for k,r in enumerate(states,1)]).mean()
                    val_loss.append(v.item()*len(indices))
            history.append({'epoch':epoch+1,'train_step_bce':float(np.mean(losses)),
                            'validation_step_bce':sum(val_loss)/len(tensors['val']['problem']),
                            'gradient_norm_max':epoch_grad})
            if epoch==0 or (epoch+1)%20==0:
                print(f'{name} epoch={epoch+1} train={np.mean(losses):.5f} val={history[-1]["validation_step_bce"]:.5f}',flush=True)
    except Exception:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoints/(name+'_FAILED.pt'))
        write(run/(name+'_history_FAILED.json'),history)
        raise
    if device.type=='cuda':
        torch.cuda.synchronize()
    duration=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
    record={'name':name,'seed':seed,'initial_sha256':initial_hash,'final_sha256':state_hash(model),
            'batch_order_sha256':order_hash.hexdigest(),'optimizer_steps':steps,'train_core_calls':8*steps,
            'own_train_seconds':duration,'train_peak_allocated_bytes':peak,
            'train_peak_incremental_bytes':peak-before_memory if peak is not None else None,
            'gradient_norm_max':max_grad,'parameters':parameters(model),
            'train_forward_linear_flops':linear_flops(model)*8*len(tensors['train']['problem'])*config['epochs_core']}
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),**record},checkpoints/(name+'.pt'))
    write(run/(name+'_history.json'),history)
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad=None
    return model,record


def execute(run,checkpoints,config,device):
    from verify import preservation,structural,verify_run
    write(run/'config.json',config)
    shutil.copytree(BASE/'src',run/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(run/'source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'source').glob('*.py')})
    original=preservation()
    write(run/'preservation_before.json',original)
    environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
                 'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                 'device':str(device),'platform':platform.platform(),'precision':'FP32','tf32':False,
                 'PYTHONNOUSERSITE':os.environ.get('PYTHONNOUSERSITE'),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if device.type=='cuda':
        environment['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,encoding='utf-8').strip()
    write(run/'environment.json',environment)
    write(run/'preflight.json',structural())
    data,generation=generate(config)
    np.savez_compressed(run/'dataset.npz',**data)
    write(run/'generation.json',generation)
    audit_result=audit(data,config)
    write(run/'dataset_audit.json',audit_result)
    for split in ('train','val','test'):
        print('AUDIT',split,audit_result[split]['size_difficulty'],flush=True)
    tensors={split:{k:torch.from_numpy(data[split+'_'+k]).to(device) for k in ('problem','distances','start')}
             for split in ('train','val','test')}
    records,cores=[],[]
    for seed in config['seeds']:
        core,core_record=fit_core(run,checkpoints,config,device,tensors,seed)
        cores.append(core_record)
        write(run/'core_summary.json',cores)
        histories={}
        cache_started=time.perf_counter()
        for split in ('train','val','test'):
            histories[split]=cache_history(core,tensors[split]['problem'],256)
            assert not histories[split].requires_grad
            np.savez_compressed(run/f'history_seed{seed}_{split}.npz',states=histories[split].cpu().numpy())
        if device.type=='cuda':
            torch.cuda.synchronize()
        cache_seconds=time.perf_counter()-cache_started
        write(run/f'cache_seed{seed}.json',{'seconds_including_compression_io':cache_seconds,'core_sha256':state_hash(core),
             'bytes':sum(h.numel()*h.element_size() for h in histories.values()),
             'sha256':{s:hashlib.sha256(h.cpu().numpy().tobytes()).hexdigest() for s,h in histories.items()},
             'source':'model-generated histories from frozen core; no BFS targets enter features',
             'cache_forward_core_calls':8*sum((len(t['problem'])+255)//256 for t in tensors.values())})
        initial=head_initial(seed)
        for mode in ('C','D','E','A','A_large','A_flops'):
            model,record=fit_logits(run,checkpoints,config,device,tensors,histories,core,seed,mode,initial if mode in ('C','D','E') else None)
            probability,logits=predict(model,tensors['test']['problem'],histories['test'] if mode in ('C','D') else None,mode)
            np.savez_compressed(run/(record['name']+'_prediction.npz'),probability=probability,logits=logits)
            record['latency']=latency(model,core,torch.from_numpy(data['test_problem'][:config['batch_size']]),device,mode,config)
            record['total_system_parameters']=parameters(model)+(parameters(core) if mode in ('C','D') else 0)
            record['total_training_steps_including_core']=record['optimizer_steps']+(core_record['optimizer_steps'] if mode in ('C','D') else 0)
            record['total_training_seconds_including_core']=record['own_train_seconds']+(core_record['own_train_seconds'] if mode in ('C','D') else 0)
            record['training_core_calls']=core_record['train_core_calls'] if mode in ('C','D') else 0
            record['cache_preparation_seconds_separate']=cache_seconds if mode in ('C','D') else 0.
            if mode=='D':
                for perturbation in INTERVENTIONS:
                    p,z=predict(model,tensors['test']['problem'],histories['test'],'D',perturbation=perturbation)
                    np.savez_compressed(run/f'D_seed{seed}_ablation_{perturbation}.npz',probability=p,logits=z)
            records.append(record)
            write(run/'summary.json',records)
            print(f'{record["name"]} complete own_train={record["own_train_seconds"]:.1f}s',flush=True)
            del model
        b={'name':f'B_seed{seed}','mode':'B','seed':seed,'core_sha256':state_hash(core),
           'total_system_parameters':parameters(core),'trainable_parameters':parameters(core),
           'optimizer_steps':core_record['optimizer_steps'],'total_training_steps_including_core':core_record['optimizer_steps'],
           'training_core_calls':core_record['train_core_calls'],'train_head_calls':0,'train_baseline_calls':0,
           'own_train_seconds':core_record['own_train_seconds'],'total_training_seconds_including_core':core_record['own_train_seconds'],
           'train_peak_allocated_bytes':core_record['train_peak_allocated_bytes'],
           'train_peak_incremental_bytes':core_record['train_peak_incremental_bytes'],
           'latency':latency(None,core,torch.from_numpy(data['test_problem'][:config['batch_size']]),device,'B',config)}
        np.savez_compressed(run/(b['name']+'_prediction.npz'),probability=histories['test'][:,-1].cpu().numpy())
        records.append(b)
        write(run/'summary.json',records)
        assert state_hash(core)==core_record['final_sha256']
        del histories,core
    from analyze import analyze
    analyze(run)
    write(run/'verification.json',verify_run(run,config,device,original))
    from visualize import render
    render(run)
    write(run/'status.json',{'status':'complete','models':len(records),'cores':len(cores)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=BASE/'configs/experiment.json')
    parser.add_argument('--device',choices=('cuda','cpu'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if os.environ.get('PYTHONNOUSERSITE')!='1':
        raise RuntimeError('Set PYTHONNOUSERSITE=1 before starting Python')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.device:
        config['device']=args.device
    config['smoke']=args.smoke
    if args.smoke:
        config.update(seeds=[42],sizes_per_maze_size=[64,32,32],epochs_core=2,epochs_heads=2,epochs_baselines=2,latency_warmup=2,latency_repeats=3)
    device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable; torch={torch.__version__}, CUDA build={torch.version.cuda}; check driver/build or use --device cpu explicitly')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S')+('_smoke_' if args.smoke else '_main_')+uuid.uuid4().hex[:8]
    run,checkpoints=BASE/'results'/run_id,BASE/'checkpoints'/run_id
    run.mkdir(parents=True,exist_ok=False)
    checkpoints.mkdir(parents=True,exist_ok=False)
    print('RUN_ID='+run_id,flush=True)
    try:
        execute(run,checkpoints,config,device)
    except Exception:
        write(run/'failure.json',{'status':'failed','traceback':traceback.format_exc()})
        raise
    print('COMPLETE '+str(run),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
