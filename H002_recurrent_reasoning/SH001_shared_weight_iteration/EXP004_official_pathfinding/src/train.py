"""Official-data entrypoint: shared core, frozen heads, baselines and actual routes."""
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
from model import Core,Head,Baseline,features,loss,state_hash,parameters,flops
from prepare_dataset import prepare,load_dataset,SPLITS
from evaluate import histories,predict,latency,evaluate_predictions,summarize,grouping,aggregate,VARIANTS

BASE=Path(__file__).resolve().parents[1]
RESUME=False


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def make_tensors(data,ids,device,scale):
    walls=torch.from_numpy(data['walls'][data['map_id'][ids]].astype(np.float32)).to(device)
    problem=torch.cat((walls,torch.zeros_like(walls),torch.zeros_like(walls)),1)
    indices=torch.arange(len(ids),device=device)
    problem[indices,1024+torch.from_numpy(data['start'][ids].astype(np.int64)).to(device)]=1
    problem[indices,2048+torch.from_numpy(data['goal'][ids].astype(np.int64)).to(device)]=1
    target=torch.from_numpy(data['fields'][data['field_id'][ids]].astype(np.float32)).to(device)
    mask=(target>=0).float()
    target=target.clamp_min(0)/scale
    return {'problem':problem,'target':target,'mask':mask}


def fit(run,checkpoints,config,device,tensors,cache,core,seed,mode,initial=None):
    torch.manual_seed(seed if mode=='core' else seed+50000)
    model=Core() if mode=='core' else Head() if mode in ('C','D','E') else Baseline(config['baseline_widths'][mode])
    if initial is not None: model.load_state_dict(initial)
    checkpoint=checkpoints/f'{mode}_seed{seed}.pt'
    if RESUME and checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(saved['model']); model=model.to(device)
        assert state_hash(model)==saved['final_hash']
        assert saved['epochs']==(config['epochs_core'] if mode=='core' else config['epochs_heads'] if mode in ('C','D','E') else config['epochs_baselines'])
        print('RESUMED '+checkpoint.name,flush=True)
        return model,{k:v for k,v in saved.items() if k not in ('model','optimizer')}
    initial_hash=state_hash(model)
    model=model.to(device)
    core_hash=state_hash(core) if core is not None else None
    if core is not None: assert all(not p.requires_grad and p.grad is None for p in core.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    epochs=config['epochs_core'] if mode=='core' else config['epochs_heads'] if mode in ('C','D','E') else config['epochs_baselines']
    generator=torch.Generator().manual_seed(seed+10000)
    order_hash=hashlib.sha256(); name=f'{mode}_seed{seed}'; history=[]; steps=0; max_grad=0.
    baseline_memory=torch.cuda.memory_allocated() if device.type=='cuda' else None
    if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    def forward(split,index):
        p=tensors[split]['problem'][index]
        if mode=='core': return model(p,8)
        h=cache[split][index] if mode in ('C','D') else None
        return model(features(p,h,mode)) if mode in ('C','D','E') else model(p)
    try:
        for epoch in range(epochs):
            model.train(); losses=[]; epoch_gradient=0.
            order=torch.randperm(len(tensors['train']['problem']),generator=generator)
            order_hash.update(order.numpy().tobytes())
            for index in order.to(device).split(config['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                output=forward('train',index)
                target,mask=[tensors['train'][k][index] for k in ('target','mask')]
                if mode=='core':
                    objectives=torch.stack([loss(r,target,mask) for r in output])
                    objective=config['core_final_loss_weight']*objectives[-1]+config['core_all_step_loss_weight']*objectives.mean()
                else: objective=loss(output,target,mask)
                if not torch.isfinite(objective): raise FloatingPointError('Nonfinite training loss')
                objective.backward()
                norm=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
                if not np.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
                epoch_gradient=max(epoch_gradient,norm)
                optimizer.step(); steps+=1; losses.append(objective.item()*len(index))
            max_grad=max(max_grad,epoch_gradient)
            model.eval(); val=0.
            with torch.no_grad():
                for index in torch.arange(len(tensors['val']['problem']),device=device).split(config['eval_batch_size']):
                    output=forward('val',index)
                    final=output[-1] if mode=='core' else output
                    v=loss(final,tensors['val']['target'][index],tensors['val']['mask'][index])
                    val+=v.item()*len(index)
            record={'epoch':epoch+1,'train_loss':sum(losses)/len(tensors['train']['problem']),
                    'validation_final_mse':val/len(tensors['val']['problem']),'gradient_norm_max':epoch_gradient}
            history.append(record)
            if epoch==0 or (epoch+1)%20==0: print(f'{name} epoch={epoch+1} train={record["train_loss"]:.6f} val={record["validation_final_mse"]:.6f}',flush=True)
    except Exception:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoints/(name+'_FAILED.pt'))
        write(run/(name+'_history_FAILED.json'),history)
        raise
    if device.type=='cuda': torch.cuda.synchronize()
    duration=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
    if core is not None:
        assert core_hash==state_hash(core) and all(p.grad is None for p in core.parameters())
    record={'name':name,'mode':mode,'seed':seed,'initial_hash':initial_hash,'final_hash':state_hash(model),
            'core_hash':core_hash,'order_hash':order_hash.hexdigest(),'epochs':epochs,'steps':steps,
            'own_train_seconds':duration,'peak_allocated_bytes':peak,'peak_incremental_bytes':peak-baseline_memory if peak is not None else None,
            'gradient_norm_max':max_grad,'parameters':parameters(model),'core_calls':8*steps if mode=='core' else 0,
            'head_calls':steps if mode in ('C','D','E') else 0,
            'training_forward_flops':flops(model)*len(tensors['train']['problem'])*epochs*(8 if mode=='core' else 1)}
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),**record},checkpoints/(name+'.pt'))
    write(run/(name+'_history.json'),history)
    return model,record


def record_evaluation(run,model,core,problem,cache,data,ids,config,device,seed,mode,metadata,rows):
    started=time.perf_counter()
    scores=predict(model,problem,cache if mode in ('B','C','D') else None,mode,config['eval_batch_size'])
    np.save(run/f'{mode}_seed{seed}_scores.npy',scores)
    values,paths=evaluate_predictions(scores,data,ids,config)
    elapsed=time.perf_counter()-started
    np.savez_compressed(run/f'{mode}_seed{seed}_sample_metrics.npz',**values)
    write(run/f'{mode}_seed{seed}_paths.json',paths)
    rows.extend(summarize(values,grouping(data,ids),mode,seed))
    metadata['evaluation_seconds_including_io_and_decoder']=elapsed
    metadata['latency']=latency(model,core,problem[:config['batch_size']].cpu(),device,mode,config)
    print(f'{mode}_seed{seed} seen={values["success"][data["split"][ids]==2].mean():.4f} unseen={values["success"][data["split"][ids]==3].mean():.4f}',flush=True)
    return scores


def execute(run,checkpoints,config,device,dataset_path):
    from verify import preservation,structural,verify_run
    original=preservation()
    if RESUME:
        assert original==json.loads((run/'preservation_before.json').read_text(encoding='utf-8'))
        assert config==json.loads((run/'config.json').read_text(encoding='utf-8'))
        for name,digest in json.loads((run/'dataset_reference.json').read_text(encoding='utf-8'))['files'].items():
            assert hashlib.sha256((dataset_path/name).read_bytes()).hexdigest()==digest
        resume_source=run/('resume_source_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        shutil.copytree(BASE/'src',resume_source,ignore=shutil.ignore_patterns('__pycache__'))
        write(resume_source/'hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in resume_source.glob('*.py')})
        write(resume_source/'resume.json',{'reason':'Execution interrupted after completed stages; restore finished checkpoints and retain recorded evaluations','at':datetime.now().isoformat()})
    else:
        write(run/'preservation_before.json',original)
    if not RESUME:
        write(run/'config.json',config)
        write(run/'dataset_reference.json',{'path':str(dataset_path),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset_path.iterdir() if p.is_file()}})
        shutil.copytree(BASE/'src',run/'source',ignore=shutil.ignore_patterns('__pycache__'))
        write(run/'source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'source').glob('*.py')})
        environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
                     'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                     'platform':platform.platform(),'device':str(device),'precision':'FP32','tf32':False,
                     'PYTHONNOUSERSITE':os.environ.get('PYTHONNOUSERSITE'),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
        if device.type=='cuda': environment['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,encoding='utf-8').strip()
        write(run/'environment.json',environment)
        write(run/'preflight.json',structural())
    data=load_dataset(dataset_path)
    split_ids={name:np.flatnonzero(data['split']==i) for i,name in enumerate(SPLITS)}
    if config['smoke']:
        split_ids={name:indices[:128 if name=='train' else 32] for name,indices in split_ids.items()}
    eval_ids=np.concatenate((split_ids['test_seen'],split_ids['test_unseen']))
    np.savez(run/'used_indices.npz',**split_ids,evaluation=eval_ids)
    tensors={name:make_tensors(data,split_ids[name],device,config['distance_scale']) for name in ('train','val')}
    tensors['evaluation']=make_tensors(data,eval_ids,device,config['distance_scale'])
    def previous(name):
        path=run/name
        return json.loads(path.read_text(encoding='utf-8')) if RESUME and path.exists() else []
    rows,ablations,summary,core_records=previous('group_results.json'),previous('ablation_results.json'),previous('summary.json'),[]
    for seed in config['seeds']:
        core,core_record=fit(run,checkpoints,config,device,tensors,None,None,seed,'core')
        core_records.append(core_record); write(run/'core_summary.json',core_records)
        for p in core.parameters(): p.requires_grad_(False); p.grad=None
        cache={}; cache_started=time.perf_counter(); cache_hashes={}
        for split in ('train','val','evaluation'):
            cache[split]=histories(core,tensors[split]['problem'],config['eval_batch_size'])
            cache_hashes[split]=hashlib.sha256(cache[split].cpu().numpy().tobytes()).hexdigest()
        history_path=run/f'core_seed{seed}_evaluation_history.npy'
        if RESUME and history_path.exists():
            assert np.array_equal(np.load(history_path,mmap_mode='r'),cache['evaluation'].cpu().numpy())
            assert cache_hashes==json.loads((run/f'cache_seed{seed}.json').read_text(encoding='utf-8'))['sha256']
        else: np.save(history_path,cache['evaluation'].cpu().numpy())
        if not (run/f'cache_seed{seed}.json').exists(): write(run/f'cache_seed{seed}.json',{'core_hash':state_hash(core),'sha256':cache_hashes,
              'bytes':sum(h.numel()*h.element_size() for h in cache.values()),'seconds_including_test_history_io':time.perf_counter()-cache_started})
        torch.manual_seed(seed+50000); initial=copy.deepcopy(Head().state_dict())
        for mode in ('C','D','E','A','A_large','A_flops'):
            if any(r['mode']==mode and r['seed']==seed for r in summary):
                print(f'RETAINED {mode}_seed{seed} evaluation',flush=True); continue
            model,record=fit(run,checkpoints,config,device,tensors,cache,core,seed,mode,initial if mode in ('C','D','E') else None)
            record['total_system_parameters']=parameters(model)+(parameters(core) if mode in ('C','D') else 0)
            record['system_train_seconds']=record['own_train_seconds']+(core_record['own_train_seconds'] if mode in ('C','D') else 0)
            record['system_steps']=record['steps']+(core_record['steps'] if mode in ('C','D') else 0)
            record['system_core_calls']=core_record['core_calls'] if mode in ('C','D') else 0
            record_evaluation(run,model,core,tensors['evaluation']['problem'],cache['evaluation'],data,eval_ids,config,device,seed,mode,record,rows)
            if mode=='D':
                checks={}
                for variant in VARIANTS:
                    scores=predict(model,tensors['evaluation']['problem'],cache['evaluation'],'D',config['eval_batch_size'],variant)
                    values,_=evaluate_predictions(scores,data,eval_ids,config)
                    np.savez_compressed(run/f'D_seed{seed}_ablation_{variant}_metrics.npz',**values)
                    checks[variant]=hashlib.sha256(scores.tobytes()).hexdigest()
                    ablations.extend(summarize(values,grouping(data,eval_ids),'D',seed,variant))
                write(run/f'D_seed{seed}_ablation_prediction_hashes.json',checks)
            summary.append(record); write(run/'summary.json',summary)
            write(run/'group_results.json',rows); write(run/'ablation_results.json',ablations)
            del model
        if any(r['mode']=='B' and r['seed']==seed for r in summary):
            del cache,core
            continue
        b={**core_record,'mode':'B','name':f'B_seed{seed}','total_system_parameters':parameters(core),
           'system_train_seconds':core_record['own_train_seconds'],'system_steps':core_record['steps'],'system_core_calls':core_record['core_calls']}
        record_evaluation(run,None,core,tensors['evaluation']['problem'],cache['evaluation'],data,eval_ids,config,device,seed,'B',b,rows)
        summary.append(b); write(run/'summary.json',summary); write(run/'group_results.json',rows)
        assert state_hash(core)==core_record['final_hash']
        del cache,core
    write(run/'aggregate.json',aggregate(rows)); write(run/'ablation_aggregate.json',aggregate(ablations))
    write(run/'verification.json',verify_run(run,config,device,original))
    from visualize import render
    render(run)
    from verify import verify_visualization
    visual_check=verify_visualization(run)
    write(run/'visualization_verification.json',visual_check)
    write(run/'status.json',{'status':'complete','models':len(summary),'dataset':str(dataset_path)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=BASE/'configs/experiment.json')
    parser.add_argument('--resume',type=Path,help='Resume an interrupted run without retraining completed checkpoints')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--device',choices=('cuda','cpu'))
    global RESUME
    args=parser.parse_args()
    RESUME=args.resume is not None
    if os.environ.get('PYTHONNOUSERSITE')!='1': raise RuntimeError('Set PYTHONNOUSERSITE=1')
    config=json.loads(((args.resume/'config.json') if RESUME else args.config).read_text(encoding='utf-8'))
    dataset_path=prepare(config)
    if RESUME: args.smoke=config['smoke']
    config['smoke']=args.smoke
    if args.smoke: config.update(seeds=[42],epochs_core=2,epochs_heads=2,epochs_baselines=2,latency_warmup=2,latency_repeats=3)
    if args.device: config['device']=args.device
    device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError(f'CUDA unavailable: torch={torch.__version__}, CUDA build={torch.version.cuda}')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S')+('_smoke_' if args.smoke else '_main_')+uuid.uuid4().hex[:8]
    run,checkpoints=BASE/'results'/run_id,BASE/'checkpoints'/run_id
    if RESUME:
        run=args.resume.resolve(); checkpoints=BASE/'checkpoints'/run.name
        assert run.is_dir() and checkpoints.is_dir() and not (run/'status.json').exists()
    else:
        run.mkdir(parents=True,exist_ok=False); checkpoints.mkdir(parents=True,exist_ok=False)
    print('RUN_ID='+run.name,flush=True)
    try: execute(run,checkpoints,config,device,dataset_path)
    except Exception:
        write(run/'failure.json',{'traceback':traceback.format_exc()}); raise
    print('COMPLETE '+str(run),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
