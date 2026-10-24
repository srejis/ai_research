"""Frozen-solver Bernoulli stopping, terminal-only cost and REINFORCE."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from common import BASE,read,write,new_run,setup,seed_all,model_hash,checkpoint_path,save_checkpoint,rng_state,restore_rng,checked_resume,preservation,sha
from model import Solver,Halt
from dataset import Dataset
from labels import policy_loss
from evaluate import evaluate,values_for,latency


def freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False); p.grad=None


def cache_states(solver,ds,ids,batch,path):
    marker=path.with_suffix('.cache.json')
    signature={'solver_hash':model_hash(solver),'problem_ids_sha256':__import__('hashlib').sha256(ids.tobytes()).hexdigest()}
    if marker.exists() and read(marker).get('complete'):
        saved=read(marker); assert all(saved[k]==v for k,v in signature.items())
        assert sha(path)==saved['sha256']
        # This cache is only a training optimization for a fully frozen deterministic solver.
        return np.load(path,mmap_mode='r')
    start_at=0
    if path.exists():
        assert marker.exists(),'Incomplete cache without marker retained; inspect explicitly'
        saved=read(marker); assert all(saved[k]==v for k,v in signature.items()); start_at=saved['next_problem']
        cache=np.lib.format.open_memmap(path,mode='r+')
    else:
        cache=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ids),16,16,128))
        write(marker,{**signature,'complete':False,'next_problem':0})
    with torch.no_grad():
        for start in range(start_at,len(ids),batch):
            x,_=ds.batch(ids[start:start+batch]); _,states,_,_=solver(x,16,True,return_states=True)
            cache[start:start+len(x)]=torch.stack(states,1).cpu().numpy()
            cache.flush(); write(marker,{**signature,'complete':False,'next_problem':start+len(x)})
    cache.flush(); del cache
    write(marker,{**signature,'complete':True,'sha256':sha(path),'next_problem':len(ids)})
    return np.load(path,mmap_mode='r')


def sampled_terminal(halt,states):
    """No all-terminal loss expectation: draw actions, read only the chosen state."""
    batch=len(states); active=torch.ones(batch,dtype=torch.bool,device=states.device)
    times=torch.zeros(batch,dtype=torch.long,device=states.device); forced=torch.zeros_like(active)
    log_probability=torch.zeros(batch,device=states.device); events=[]
    for t in range(16):
        idx=active.nonzero().squeeze(-1)
        if not len(idx): break
        logits=halt(states[idx,t]); distribution=torch.distributions.Bernoulli(logits=logits)
        action=distribution.sample(); lp=distribution.log_prob(action)
        log_probability=log_probability.index_add(0,idx,lp)
        stop=action.bool(); end=idx if t==15 else idx[stop]
        times[end]=t+1; active[end]=False
        if t==15: forced[idx]=~stop
        events.append({'t':t+1,'ids_in_batch':idx.detach().cpu().tolist(),'halt_probability':logits.sigmoid().detach().cpu().tolist(),
                       'action':action.detach().cpu().tolist(),'log_probability':lp.detach().cpu().tolist()})
    selected=states[torch.arange(batch,device=states.device),times-1]
    return selected,times,forced,log_probability,events


def maintains(candidate,baseline):
    # Zero observed accuracy/SPL sacrifice, including all nonempty validation difficulty bins.
    for group,reference in baseline.items():
        if group!='val' and not group.startswith('val/Q'): continue
        if not reference['n']: continue
        if candidate[group]['success']+1e-12<reference['success'] or candidate[group]['spl']+1e-12<reference['spl']: return False
    return True


def train(solver_run,config,resume=None,smoke=False,on_started=None):
    if smoke: config={**config,'halt_epochs':2,'halt_costs':[0.,.01],'seeds':[42],'latency_repeats':3}
    run=resume or new_run('halt_smoke' if smoke else 'halt',config)
    if resume: checked_resume(run,config)
    else:
        for name in ('data_hashes.json','preservation_before.json','dataset_audit.json'): write(run/name,read(solver_run/name))
        write(run/'solver_reference.json',{'run':str(solver_run.resolve()),'seeds':config['seeds']})
    ds=Dataset('cuda'); ids=dict(np.load(solver_run/'used_problem_ids.npz')); summaries=[]
    if on_started: on_started(run)
    for seed in config['seeds']:
        solver=Solver().to(ds.device)
        source=torch.load(checkpoint_path(solver_run,f'B_seed{seed}_best'),map_location=ds.device,weights_only=False)
        solver.load_state_dict(source['model']); freeze(solver); frozen=model_hash(solver)
        baseline,_,_=evaluate(solver,ds,ids['val'],16,True,config['eval_batch'])
        baseline_latency=latency(solver,ds,ids['val'],16,True,repeats=config['latency_repeats'])
        cache_path=run/f'B_seed{seed}_train_states.npy'; cache_started=time.perf_counter()
        cached=cache_states(solver,ds,ids['train'],config['eval_batch'],cache_path)
        write(run/f'B_seed{seed}_cache.json',{'shape':list(cached.shape),'bytes':cached.nbytes,'seconds':time.perf_counter()-cache_started,
                                          'solver_hash':frozen,'role':'frozen training-state cache only; actual adaptive inference compacts active examples'})
        candidates=[]
        for penalty in config['halt_costs']:
            tag=f'seed{seed}_cost{penalty:g}'; record_path=run/(tag+'_result.json')
            if record_path.exists(): candidates.append(read(record_path)); continue
            seed_all(seed+70000); halt=Halt().to(ds.device)
            # Same mild late-stop prior for all cost candidates; correctness learned first.
            halt.net[-1].bias.data.fill_(-2.)
            if penalty>0:
                # Accuracy is learned with zero compute cost first; positive-cost runs branch from it.
                zero=torch.load(checkpoint_path(run,f'seed{seed}_cost0_best'),map_location=ds.device,weights_only=False)
                halt.load_state_dict(zero['halt'])
            initial_halt_hash=model_hash(halt)
            optimizer=torch.optim.AdamW(halt.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            running=3.; start_epoch=0; history=[]; best=None; elapsed=0.
            last=checkpoint_path(run,tag+'_last')
            if resume and last.exists():
                saved=torch.load(last,map_location=ds.device,weights_only=False)
                halt.load_state_dict(saved['halt']); optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
                running=saved['baseline']; start_epoch=saved['epoch']; history=saved['history']; best=saved['best_key']; elapsed=saved['elapsed']
            started=time.perf_counter()
            event_file=(run/(tag+'_actions.jsonl')).open('a' if start_epoch else 'w',encoding='utf-8')
            try:
                for epoch in range(start_epoch,config['halt_epochs']):
                    halt.train(); costs=[]; pg_values=[]; step_values=[]; success_values=[]
                    order=np.random.default_rng(seed+90000+epoch).permutation(len(ids['train']))
                    for batch_number,index in enumerate(np.array_split(order,int(np.ceil(len(order)/config['batch_size'])))):
                        states=torch.from_numpy(np.array(cached[index],copy=True)).to(ds.device)
                        terminal,times,forced,logp,events=sampled_terminal(halt,states)
                        problem_ids=ids['train'][index]; x,y=ds.batch(problem_ids)
                        with torch.no_grad():
                            # Exactly one final readout at the actually sampled termination time.
                            output=solver.readout(x,terminal)
                            losses=policy_loss(output,y)
                            metrics,_=values_for(output.cpu().numpy(),ds.labels[ds.data['field_id'][problem_ids]],ds.data,problem_ids,
                                                 times.cpu().numpy(),forced.cpu().numpy())
                            success=torch.as_tensor(metrics['success'],dtype=torch.float32,device=ds.device)
                            cost=losses+2*(1-success)+penalty*times/16
                        previous_baseline=running
                        objective=((cost-previous_baseline).detach()*logp).mean()
                        if not torch.isfinite(objective): raise FloatingPointError('Nonfinite REINFORCE objective')
                        optimizer.zero_grad(set_to_none=True); objective.backward(); torch.nn.utils.clip_grad_norm_(halt.parameters(),1.); optimizer.step()
                        running=.95*running+.05*float(cost.mean())
                        costs.extend(cost.cpu().tolist()); step_values.extend(times.cpu().tolist()); success_values.extend(metrics['success'].tolist()); pg_values.append(float(objective.detach()))
                        import json
                        event_file.write(json.dumps({'epoch':epoch+1,'batch':batch_number,'problem_ids':problem_ids.tolist(),'events':events,
                            'terminal_step':times.cpu().tolist(),'forced':forced.cpu().tolist(),'cost':cost.cpu().tolist(),
                            'baseline_before_action':previous_baseline,'logp_sum':logp.detach().cpu().tolist(),'policy_loss':float(objective.detach()),
                            'terminal_policy_losses':losses.cpu().tolist(),'terminal_success':metrics['success'].tolist()})+'\n')
                    event_file.flush(); halt.eval()
                    groups,_,_=evaluate(solver,ds,ids['val'],16,True,config['eval_batch'],halt=halt)
                    qualifies=maintains(groups,baseline)
                    # Prefer feasible policies, then fewer steps; infeasible selection preserves accuracy first.
                    candidate_key=(int(qualifies),-groups['val']['steps'] if qualifies else groups['val']['success'],
                                   groups['val']['spl'],-groups['val']['loss'])
                    improved=best is None or candidate_key>tuple(best)
                    if improved: best=candidate_key
                    row={'epoch':epoch+1,'cost':float(np.mean(costs)),'reinforce_loss':float(np.mean(pg_values)),
                         'sampled_mean_steps':float(np.mean(step_values)),'sampled_success':float(np.mean(success_values)),
                         'validation':groups,'quality_qualified':qualifies}; history.append(row)
                    saved={'halt':halt.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),'epoch':epoch+1,
                           'baseline':running,'history':history,'best_key':best,'elapsed':elapsed+time.perf_counter()-started,
                           'solver_hash':frozen,'config':config,'source':read(run/'source_hashes.json')}
                    save_checkpoint(last,saved)
                    if improved: save_checkpoint(checkpoint_path(run,tag+'_best'),saved)
                    write(run/(tag+'_history.json'),history)
                    print(f'HALT {tag} epoch={epoch+1} T={groups["val"]["steps"]:.2f} success={groups["val"]["success"]:.5f} feasible={qualifies}',flush=True)
            finally: event_file.close()
            training_seconds=elapsed+time.perf_counter()-started
            saved=torch.load(checkpoint_path(run,tag+'_best'),map_location=ds.device,weights_only=False); halt.load_state_dict(saved['halt']); halt.eval()
            groups,metrics,paths=evaluate(solver,ds,ids['evaluation'],16,True,config['eval_batch'],halt=halt,save=run/tag)
            val_groups,_,_=evaluate(solver,ds,ids['val'],16,True,config['eval_batch'],halt=halt)
            timing=latency(solver,ds,ids['val'],16,True,halt=halt,repeats=config['latency_repeats'])
            qualifies=maintains(val_groups,baseline)
            faster=timing[0]['mean_ms']<baseline_latency[0]['mean_ms']
            record={'seed':seed,'penalty':penalty,'groups':groups,'validation':val_groups,'baseline_validation':baseline,
                    'baseline_latency':baseline_latency,'latency':timing,'quality_qualified':qualifies,
                    'adoptable':qualifies and baseline['val']['success']>0 and val_groups['val']['steps']<16 and faster,
                    'nonzero_solver_validation_success':baseline['val']['success']>0,'halt_hash':model_hash(halt),
                    'solver_hash':frozen,'train_seconds':training_seconds,'selected_epoch':saved['epoch'],
                    'candidate_stage_seconds_before_replay':elapsed+time.perf_counter()-started}
            record.update(initial_halt_hash=initial_halt_hash,warm_start_from_zero_cost=penalty>0)
            record['system_halt_train_seconds']=training_seconds+(read(run/f'seed{seed}_cost0_result.json')['train_seconds'] if penalty>0 else 0.)
            # Restore the saved head and reproduce deterministic actual adaptive routes.
            restored=Halt().to(ds.device); restored.load_state_dict(saved['halt']); restored.eval()
            _,again,replayed=evaluate(solver,ds,ids['evaluation'],16,True,config['eval_batch'],halt=restored,
                                     expected_hash=read(run/(tag+'_prediction_hash.json'))['sha256'])
            assert paths==replayed and all(np.array_equal(metrics[k],again[k],equal_nan=True) for k in metrics)
            assert model_hash(solver)==frozen and all(p.grad is None for p in solver.parameters())
            record['restored_routes_metrics_exact']=True; record['solver_frozen_exact']=True
            write(record_path,record); candidates.append(record)
        feasible=[r for r in candidates if r['adoptable']]
        chosen=min(feasible,key=lambda r:(r['validation']['val']['steps'],r['latency'][0]['mean_ms'])) if feasible else None
        selection={'seed':seed,'adopted_penalty':chosen['penalty'] if chosen else None,'fallback':'fixed16' if chosen is None else None,
                   'selection_uses_test':False,'reason':'zero observed overall/Q-bin validation success and SPL drop plus actual batch1 speedup required'}
        summaries.append({'selection':selection,'candidates':candidates}); write(run/'summary.json',summaries)
    assert preservation()==read(run/'preservation_before.json')
    write(run/'verification.json',{'preservation_exact':True,'frozen_solver_exact':True,'all_candidates_restored':True})
    write(run/'complete.json',{'phase':'halt_smoke' if smoke else 'halt','seeds':len(summaries)})
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('solver_run',type=Path); parser.add_argument('--resume',type=Path); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); setup(); config=read(args.resume/'config.json' if args.resume else BASE/'configs/experiment.json')
    try: train(args.solver_run,config,args.resume,args.smoke)
    except BaseException:
        # The redirected stderr and epoch checkpoint retain failure evidence.
        write(BASE/'results/halt_last_failure.json',{'traceback':traceback.format_exc(),'solver_run':str(args.solver_run)})
        raise
