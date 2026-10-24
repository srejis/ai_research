"""Final-only solver training, explicit phases, atomic epoch/RNG checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import hashlib
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from common import BASE,DATA,read,write,new_run,setup,seed_all,model_hash,checkpoint_path,save_checkpoint,rng_state,restore_rng,checked_resume,source_hashes,preservation
from dataset import Dataset
from labels import policy_loss
from model import Solver,parameters,estimated_flops
from evaluate import evaluate,latency


def condition(mode): return (1,False) if mode=='A' else (16,mode=='B')


def validation(model,ds,ids,steps,history,batch):
    rows,_,_=evaluate(model,ds,ids,steps,history,batch)
    name='val' if 'val' in rows else next(iter(rows))
    return rows[name],rows


def key(row): return (row['success'],row['spl'],-row['loss'])


def fit(run,config,ds,seed,mode,train_ids,val_ids,resume=False):
    name=f'{mode}_seed{seed}'; completed=run/(name+'_training.json')
    steps,history=condition(mode)
    if completed.exists():
        record=read(completed); saved=torch.load(checkpoint_path(run,name+'_best'),map_location=ds.device,weights_only=False)
        model=Solver().to(ds.device); model.load_state_dict(saved['model'])
        assert model_hash(model)==saved['model_hash']; print('RETAINED '+name,flush=True)
        return model,record
    seed_all(seed); model=Solver().to(ds.device); initial=model_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    last=checkpoint_path(run,name+'_last'); records=[]; best_key=None; start_epoch=0; count=0; elapsed=0.
    if resume and last.exists():
        saved=torch.load(last,map_location=ds.device,weights_only=False)
        assert saved['config']==config and saved['source']==read(run/'source_hashes.json')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
        records=saved['history']; start_epoch=saved['epoch']; best_key=tuple(saved['best_key']); count=saved['optimizer_steps']; elapsed=saved['elapsed_seconds']
        initial=saved['initial_hash']; print(f'RESUME {name} epoch={start_epoch}',flush=True)
    initial_memory=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter(); orders=hashlib.sha256()
    try:
        for epoch in range(start_epoch,config['epochs']):
            model.train(); total=0.; largest_grad=0.
            order=np.random.default_rng(seed+100000+epoch).permutation(train_ids); orders.update(order.tobytes())
            for part in np.array_split(order,int(np.ceil(len(order)/config['batch_size']))):
                x,y=ds.batch(part); optimizer.zero_grad(set_to_none=True)
                logits=model(x,steps,history); loss=policy_loss(logits,y).mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite final-only solver loss')
                loss.backward(); gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_gradient'])
                if not torch.isfinite(gradient): raise FloatingPointError('Nonfinite solver gradient')
                # Scalar logging only; the recurrent/history graph is never detached.
                optimizer.step(); count+=1; total+=float(loss.detach())*len(part); largest_grad=max(largest_grad,float(gradient))
            row={'epoch':epoch+1,'train_loss':total/len(train_ids),'max_unclipped_gradient':largest_grad}
            selected=(epoch+1)%config['validation_every_epochs']==0 or epoch==0 or epoch+1==config['epochs']
            if selected:
                val,_=validation(model,ds,val_ids,steps,history,config['eval_batch']); row['validation']=val
                improved=best_key is None or key(val)>best_key
                if improved: best_key=key(val)
            records.append(row)
            metadata={'epoch':epoch+1,'history':records,'config':config,'source':read(run/'source_hashes.json'),
                      'initial_hash':initial,'best_key':best_key,'optimizer_steps':count,
                      'elapsed_seconds':elapsed+time.perf_counter()-started,'rng':rng_state(),
                      'model':model.state_dict(),'model_hash':model_hash(model),'optimizer':optimizer.state_dict()}
            save_checkpoint(last,metadata)
            if selected and improved: save_checkpoint(checkpoint_path(run,name+'_best'),metadata)
            write(run/(name+'_history.json'),records)
            print(f'{name} epoch={epoch+1} train={row["train_loss"]:.6f}'+(f' val_success={val["success"]:.5f} val_loss={val["loss"]:.5f}' if selected else ''),flush=True)
        torch.cuda.synchronize()
        best=torch.load(checkpoint_path(run,name+'_best'),map_location=ds.device,weights_only=False)
        model.load_state_dict(best['model'])
        result={'mode':mode,'seed':seed,'initial_hash':initial,'best_hash':model_hash(model),'selected_epoch':best['epoch'],
                'optimizer_steps':count,'epochs':config['epochs'],'own_train_seconds':elapsed+time.perf_counter()-started,
                'parameters':parameters(model),'core_train_batch_calls':count*steps,'head_train_batch_calls':count,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_increment_bytes':torch.cuda.max_memory_allocated()-initial_memory,
                'forward_flops_per_sample':estimated_flops(steps,history),'batch_order_rule':'numpy.default_rng(seed+100000+epoch).permutation(same_train_ids)',
                'final_loss_only':True,'steps':steps,'reads_history':history}
        write(completed,result)
        return model,result
    except BaseException:
        write(run/(name+'_interruption.json'),{'traceback':traceback.format_exc(),'last_complete_epoch':records[-1]['epoch'] if records else 0,
                                               'resume':'atomic last checkpoint contains optimizer and all RNG states; incomplete epoch repeats'})
        raise


def overfit(config,ds,phase0,previous=None):
    run=new_run('overfit',config)
    write(run/'data_hashes.json',read(phase0/'data_hashes.json'))
    rng=np.random.default_rng(42); ids=rng.choice(ds.ids['train'],config['preflight_overfit_samples'],replace=False)
    np.save(run/'official_problem_ids.npy',ids)
    if previous:
        assert np.array_equal(ids,np.load(previous/'official_problem_ids.npy'))
        write(run/'extension.json',{'previous_run':str(previous),'reason':'16-problem overfit failed strict 90% completion and loss<0.15 at 2000 steps; extend optimization only, preserve architecture/labels/objective',
                                     'max_steps':config['preflight_overfit_max_steps']})
    results={}
    for mode in ('A','B'):
        seed_all(42); model=Solver().to(ds.device); opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        x,y=ds.batch(ids); steps,history=condition(mode); records=[]; started=time.perf_counter(); initial_step=0
        if previous:
            saved=torch.load(checkpoint_path(previous,mode),map_location=ds.device,weights_only=False)
            model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
            records=saved['records']; initial_step=records[-1]['step']
        for step in range(initial_step,config['preflight_overfit_max_steps']+1):
            if step==0 or step%100==0:
                model.eval()
                with torch.no_grad(): value=policy_loss(model(x,steps,history),y).mean().item()
                rows,metrics,paths=evaluate(model,ds,ids,steps,history,len(ids)); success=float(metrics['success'].mean())
                row={'step':step,'loss':value,'success':success}; records.append(row)
                print(f'OVERFIT {mode} '+str(row),flush=True); write(run/(mode+'_history.json'),records)
                if success>=.9 and value<.15: break
                save_checkpoint(checkpoint_path(run,mode+'_last'),{'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'records':records,'ids':ids})
            if step==config['preflight_overfit_max_steps']: break
            model.train(); opt.zero_grad(set_to_none=True); value=policy_loss(model(x,steps,history),y).mean(); value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        save_checkpoint(checkpoint_path(run,mode),{'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'records':records,'ids':ids})
        evaluate(model,ds,ids,steps,history,len(ids),save=run/mode)
        results[mode]={'initial_loss':records[0]['loss'],'final_loss':records[-1]['loss'],'final_success':records[-1]['success'],
                       'steps':step,'seconds':time.perf_counter()-started,'strict_pass':records[-1]['success']>=.9 and records[-1]['loss']<.15}
    write(run/'result.json',results)
    write(run/'complete.json',{'phase':'overfit','strict_pass':all(r['strict_pass'] for r in results.values())})
    return run,results


def train(config,phase0,resume=None,smoke=False,on_started=None):
    ds=Dataset('cuda')
    if smoke: config=copy.deepcopy(config); config.update(seeds=[42],epochs=2,validation_every_epochs=1,latency_repeats=3)
    run=resume or new_run('smoke' if smoke else 'solver',config)
    if resume: checked_resume(run,config)
    else:
        write(run/'data_hashes.json',read(phase0/'data_hashes.json'))
        write(run/'preservation_before.json',read(phase0/'preservation_before.json'))
        write(run/'dataset_audit.json',read(phase0/'dataset_audit.json'))
    ids={k:(v[:64] if smoke else v) for k,v in ds.ids.items()}; eval_ids=np.concatenate((ids['seen'],ids['unseen']))
    np.savez(run/'used_problem_ids.npz',**ids,evaluation=eval_ids)
    if on_started: on_started(run)
    summary=[]
    for seed in config['seeds']:
        for mode in config['models']:
            model,record=fit(run,config,ds,seed,mode,ids['train'],ids['val'],resume=resume is not None)
            steps,history=condition(mode); prefix=run/f'{mode}_seed{seed}'
            if not Path(str(prefix)+'_evaluation.json').exists():
                rows,metrics,paths=evaluate(model,ds,eval_ids,steps,history,config['eval_batch'],save=prefix)
                timing=latency(model,ds,eval_ids,steps,history,repeats=config['latency_repeats'])
                write(str(prefix)+'_evaluation.json',{'groups':rows,'latency':timing})
            record.update(read(str(prefix)+'_evaluation.json')); summary.append(record); write(run/'summary.json',summary)
            print(f'EVALUATED {mode}_seed{seed} seen={record["groups"]["seen"]["success"]:.6f} unseen={record["groups"]["unseen"]["success"]:.6f}',flush=True)
            del model
    # Replay every selected checkpoint and compare complete predictions and actual paths.
    replay={}
    for r in summary:
        name=f'{r["mode"]}_seed{r["seed"]}'; saved=torch.load(checkpoint_path(run,name+'_best'),map_location=ds.device,weights_only=False)
        model=Solver().to(ds.device); model.load_state_dict(saved['model']); steps,history=condition(r['mode'])
        replay_prefix=run/(name+'_replay')
        rows,metrics,paths=evaluate(model,ds,eval_ids,steps,history,config['eval_batch'],expected_hash=read(run/(name+'_prediction_hash.json'))['sha256'])
        old=dict(np.load(run/(name+'_metrics.npz')))
        assert all(np.array_equal(old[k],v,equal_nan=True) for k,v in metrics.items())
        assert paths==read(run/(name+'_paths.json'))
        assert model_hash(model)==r['best_hash']; replay[name]={'paths_metrics_exact':True,'logits_hash_exact':True,'checkpoint_hash_exact':True}
    assert preservation()==read(run/'preservation_before.json')
    write(run/'verification.json',{'checkpoint_replay':replay,'preservation_exact':True,'preserved_files':len(read(run/'preservation_before.json'))})
    write(run/'complete.json',{'phase':'smoke' if smoke else 'solver','models':len(summary)})
    return run


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase0',type=Path,required=True); parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--overfit',action='store_true'); parser.add_argument('--resume',type=Path)
    parser.add_argument('--extend-overfit',type=Path); parser.add_argument('--overfit-max-steps',type=int)
    args=parser.parse_args(); setup(); config=read(args.resume/'config.json' if args.resume else BASE/'configs/experiment.json')
    if args.overfit_max_steps: config={**config,'preflight_overfit_max_steps':args.overfit_max_steps}
    assert read(args.phase0/'dataset_audit.json')['oracle_optimal_rate']==1.
    if args.overfit or args.extend_overfit: overfit(config,Dataset('cuda'),args.phase0,args.extend_overfit)
    else: train(config,args.phase0,args.resume,args.smoke)


if __name__=='__main__': main()
