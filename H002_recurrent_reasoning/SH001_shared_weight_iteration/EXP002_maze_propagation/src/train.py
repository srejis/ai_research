"""One command: unique run, data audit, paired A/B/C/D training, evaluation and QA."""
import argparse
import copy
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
from torch.nn import functional as F
from maze_generator import generate,audit
from model import MazeMLP
from evaluate import evaluate,latency
from verify import preservation,structural,bfs_checks,verify_run

BASE = Path(__file__).resolve().parents[1]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def execute(run,checkpoints,config,device):
    write(run/'config.json',config)
    shutil.copytree(BASE/'src',run/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(run/'source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run/'source').glob('*.py'))})
    original = preservation()
    write(run/'preservation_before.json',original)
    environment = {'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
                   'numpy':np.__version__,'cuda_build':torch.version.cuda,'device':str(device),
                   'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                   'platform':platform.platform(),'PYTHONNOUSERSITE':os.environ.get('PYTHONNOUSERSITE'),
                   'precision':'FP32','tf32':False,'deterministic_algorithms':True,
                   'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if device.type=='cuda':
        environment['nvidia_smi'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,encoding='utf-8').strip()
    write(run/'environment.json',environment)
    write(run/'preflight.json',{'structure':structural(),'bfs':bfs_checks()})
    data,generation = generate(config)
    np.savez_compressed(run/'dataset.npz',**data)
    write(run/'generation.json',generation)
    data_audit = audit(data)
    write(run/'dataset_audit.json',data_audit)
    for split in ('train','val','test'):
        record = data_audit[split]
        print(f'AUDIT {split}: n={record["n"]} ecc_mean={record["eccentricity"]["mean"]:.2f} ecc>16={record["fraction_eccentricity_gt16"]:.3f} all-open IoU={record["all_open_baseline"]["macro_iou"]:.3f}',flush=True)
    tensors = {split:{k:torch.from_numpy(data[split+'_'+k]).to(device)
                      for k in ('walls','goal','distances','start')} for split in ('train','val','test')}
    targets = {split:(tensors[split]['distances']>=0).float() for split in tensors}
    step_targets = {split:torch.stack([((tensors[split]['distances']>=0)&(tensors[split]['distances']<=k)).float()
                                      for k in range(1,config['train_steps']+1)]) for split in tensors}
    summary = []
    for seed in config['seeds']:
        torch.manual_seed(seed)
        initial = copy.deepcopy(MazeMLP().state_dict())
        initial_hash = hashlib.sha256(b''.join(t.numpy().tobytes() for t in initial.values())).hexdigest()
        for mode in 'ABCD':
            name = f'{mode}_seed{seed}'
            model = MazeMLP().to(device)
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            generator = torch.Generator().manual_seed(seed+10000)
            order_hash = hashlib.sha256()
            history,optimizer_steps,maximum_grad,first_head_grad = [],0,0.,None
            train_calls = 1 if mode=='A' else config['train_steps']
            if device.type=='cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                for epoch in range(config['epochs']):
                    model.train()
                    host_order = torch.randperm(len(targets['train']),generator=generator)
                    order_hash.update(host_order.numpy().tobytes())
                    order = host_order.to(device)
                    losses,grad_max = [],0.
                    for indices in order.split(config['batch_size']):
                        optimizer.zero_grad(set_to_none=True)
                        states = model(tensors['train']['walls'][indices],tensors['train']['goal'][indices],train_calls,mode)
                        if not all(torch.isfinite(h).all() for h in states):
                            raise FloatingPointError('Nonfinite training states')
                        if mode=='D':
                            loss = torch.stack([F.binary_cross_entropy(h[:,128:],step_targets['train'][k,indices])
                                                for k,h in enumerate(states)]).mean()
                        else:
                            loss = F.binary_cross_entropy(states[-1][:,128:],targets['train'][indices])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite training loss')
                        loss.backward()
                        norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
                        if not np.isfinite(norm):
                            raise FloatingPointError('Nonfinite gradients')
                        if first_head_grad is None:
                            first_head_grad = model.core[2].weight.grad[:128].abs().sum().item()
                        grad_max = max(grad_max,norm)
                        optimizer.step()
                        optimizer_steps += 1
                        losses.append(loss.item())
                    maximum_grad = max(maximum_grad,grad_max)
                    model.eval()
                    with torch.no_grad():
                        val = model(tensors['val']['walls'],tensors['val']['goal'],train_calls,mode)[-1][:,128:]
                        target = step_targets['val'][-1] if mode=='D' else targets['val']
                        val_loss = F.binary_cross_entropy(val,target).item()
                        val_final = F.binary_cross_entropy(val,targets['val']).item()
                    history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),
                                    'validation_objective_last_step_bce':val_loss,'validation_final_bce':val_final,
                                    'gradient_norm_max':grad_max})
                    if epoch==0 or (epoch+1)%20==0:
                        print(f'{name} epoch={epoch+1} train={np.mean(losses):.5f} val_objective={val_loss:.5f} val_final={val_final:.5f}',flush=True)
            except Exception:
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoints/(name+'_FAILED.pt'))
                write(run/(name+'_failure.json'),{'history':history,'optimizer_steps':optimizer_steps,'traceback':traceback.format_exc()})
                raise
            if device.type=='cuda':
                torch.cuda.synchronize()
            duration = time.perf_counter()-started
            peak = torch.cuda.max_memory_allocated() if device.type=='cuda' else None
            saved = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'mode':mode,'seed':seed,
                     'config':config,'initial_sha256':initial_hash,'batch_order_sha256':order_hash.hexdigest(),
                     'optimizer_steps':optimizer_steps}
            torch.save(saved,checkpoints/(name+'.pt'))
            write(run/(name+'_history.json'),history)
            rows,states = evaluate(model,tensors['test'],mode,config['eval_steps'])
            np.savez_compressed(run/(name+'_states.npz'),states=states)
            write(run/(name+'_metrics.json'),rows)
            timings = latency(model,torch.from_numpy(data['test_walls'][:config['batch_size']]),
                              torch.from_numpy(data['test_goal'][:config['batch_size']]),device,mode,config)
            write(run/(name+'_latency.json'),timings)
            summary.append({'name':name,'mode':mode,'seed':seed,'initial_sha256':initial_hash,
                            'batch_order_sha256':order_hash.hexdigest(),'optimizer_steps':optimizer_steps,
                            'train_mapping_calls':optimizer_steps*train_calls,
                            'train_seconds_including_validation':duration,'peak_gpu_allocated_bytes':peak,
                            'gradient_norm_max':maximum_grad,'first_batch_wg_head_gradient_l1':first_head_grad,
                            'nominal_parameters':98752,'loss_connected_parameters':98752 if mode=='B' else 65856,
                            'training_forward_linear_flops':2*(192*256+256*192)*len(targets['train'])*config['epochs']*train_calls,
                            'metrics':rows,'latency':timings})
            write(run/'summary.json',summary)
            print(f'{name} complete {duration:.1f}s IoU k1={rows[0]["macro_iou"]:.4f} k8={rows[7]["macro_iou"]:.4f} k32={rows[31]["macro_iou"]:.4f}',flush=True)
    write(run/'verification.json',verify_run(run,config,device,original))
    from visualize import render
    render(run)
    write(run/'status.json',{'status':'complete','models':len(summary)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=BASE/'configs/experiment.json')
    parser.add_argument('--device',choices=('cuda','cpu'))
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if os.environ.get('PYTHONNOUSERSITE')!='1':
        raise RuntimeError('Set PYTHONNOUSERSITE=1 before starting Python')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.device:
        config['device']=args.device
    config['smoke']=args.smoke
    if args.smoke:
        config.update(seeds=[42],sizes=[128,64,64],epochs=2,latency_warmup=1,latency_repeats=2)
    device = torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable: PyTorch={torch.__version__}, CUDA build={torch.version.cuda}; check driver/CUDA PyTorch build, or explicitly use --device cpu')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')+('_smoke_' if args.smoke else '_main_')+uuid.uuid4().hex[:8]
    run,checkpoints = BASE/'results'/run_id,BASE/'checkpoints'/run_id
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
