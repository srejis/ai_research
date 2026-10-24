"""Post-training runtime review; immutable weights, exact logits, representative validation timing.

This tool is deliberately outside src/: training source versions stay frozen while training runs.
Run only after the original workflow has completed, with the reviewed inference implementation.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import copy
import hashlib
import shutil
import time
import numpy as np
import torch
from common import BASE,DATA,read,write,new_run,setup,checkpoint_path,model_hash,preservation,sha
from dataset import Dataset
from model import Solver,Halt,estimated_flops
from train_halt import freeze,maintains
from evaluate import evaluate
from visualize import render
from report import report


def representative_latency(solver,halt,ds,ids):
    rng=np.random.default_rng(20261024)
    stratified=[]
    for band in range(4):
        available=ids[ds.data['band'][ids]==band]
        stratified.extend(rng.choice(available,min(64,len(available)),replace=False).tolist())
    chosen=np.asarray(stratified,dtype=int); rng.shuffle(chosen)
    rows=[]
    with torch.no_grad():
        for batch,trials in ((1,32),(16,24),(64,20)):
            pools=[]
            for t in range(trials):
                indices=chosen[(np.arange(batch)+batch*t)%len(chosen)]
                x,_=ds.batch(indices); pools.append(x)
            def fixed(x): return solver(x,16,True)
            def adaptive(x): return solver.adaptive(x,halt)
            for _ in range(5): fixed(pools[0]); adaptive(pools[0])
            fixed_times=[]; adaptive_times=[]; actual_steps=[]
            base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            for trial,x in enumerate(pools):
                # Alternate order to limit clock/thermal ordering effects.
                funcs=[('fixed',fixed),('adaptive',adaptive)] if trial%2==0 else [('adaptive',adaptive),('fixed',fixed)]
                for name,fn in funcs:
                    torch.cuda.synchronize(); start=time.perf_counter(); measured_output=fn(x); torch.cuda.synchronize()
                    (fixed_times if name=='fixed' else adaptive_times).append((time.perf_counter()-start)*1000)
                    if name=='adaptive': actual_steps.extend(measured_output[1]['steps'].cpu().tolist())
            peak=torch.cuda.max_memory_allocated()-base
            def stats(values): return {'mean_ms':float(np.mean(values)),'std_ms':float(np.std(values)),'max_ms':max(values)}
            rows.append({'batch':batch,'trials':trials,'fixed16':stats(fixed_times),'adaptive':stats(adaptive_times),
                         'paired_mean_difference_ms':float(np.mean(np.asarray(adaptive_times)-np.asarray(fixed_times))),
                         'observed_mean_iterations_in_timing_cases':float(np.mean(actual_steps)),
                         'peak_increment_bytes_for_pair':peak,'validation_problem_pool':chosen.tolist(),
                         'scope':'GPU-resident model only; no decoder; varied official validation problems, training-defined Q1-Q4 stratified'})
    return rows


def review(workflow):
    import inspect
    assert inspect.getsource(Solver.adaptive).count('halt(h)')==1,'Apply one-halt-logit-per-step runtime fix before this review'
    state=read(workflow); assert state['status']=='complete'
    solver_run=Path(state['solver']); halt_run=Path(state['halt']); config=read(halt_run/'config.json')
    run=new_run('runtime_review',config)
    shutil.copyfile(__file__,run/'source/tools_review_runtime.py')
    hashes=read(run/'source_hashes.json'); hashes['tools_review_runtime.py']=sha(__file__); write(run/'source_hashes.json',hashes)
    for name in ('data_hashes.json','preservation_before.json','dataset_audit.json'):
        shutil.copyfile(halt_run/name,run/name)
    write(run/'review_provenance.json',{'solver_run':str(solver_run),'halt_training_run':str(halt_run),
        'reason':'Reuse one halt-logit computation per step and validate latency across varied validation cases; no retraining or policy/label/decoder change',
        'original_source_sha256':read(halt_run/'source_hashes.json'),'runtime_source_sha256':hashes,
        'selection_rule':'unchanged zero observed validation/Q success and SPL drop, nonzero baseline success, lower mean T and measured batch1 latency',
        'test_used_for_selection':False})
    for name in ('SUMMARY.json','REPORT.md'):
        if (BASE/name).exists(): shutil.copyfile(BASE/name,run/('initial_'+name))
    ds=Dataset('cuda'); ids=dict(np.load(solver_run/'used_problem_ids.npz'))
    old=read(halt_run/'summary.json'); revised=[]; replays={}; cost_rows=[]
    for seed_group in old:
        seed=seed_group['selection']['seed']; solver=Solver().to(ds.device)
        saved=torch.load(checkpoint_path(solver_run,f'B_seed{seed}_best'),map_location=ds.device,weights_only=False)
        solver.load_state_dict(saved['model']); freeze(solver); frozen=model_hash(solver)
        candidates=[]
        for original in seed_group['candidates']:
            record=copy.deepcopy(original); tag=f'seed{seed}_cost{record["penalty"]:g}'
            head_file=checkpoint_path(halt_run,tag+'_best'); saved_head=torch.load(head_file,map_location=ds.device,weights_only=False)
            halt=Halt().to(ds.device); halt.load_state_dict(saved_head['halt']); halt.eval()
            assert model_hash(halt)==record['halt_hash'] and frozen==record['solver_hash']
            head_calls=[]; core_calls=[]
            hook=halt.register_forward_hook(lambda module,args,out:head_calls.append(len(args[0])))
            core_hook=solver.core.register_forward_hook(lambda module,args,out:core_calls.append(len(args[0])))
            with torch.no_grad():
                x,_=ds.batch(ids['val'][:2]); solver.adaptive(x,halt)
            hook.remove(); core_hook.remove(); assert head_calls==core_calls
            groups,metrics,paths=evaluate(solver,ds,ids['evaluation'],16,True,config['eval_batch'],halt=halt,
                expected_hash=read(halt_run/(tag+'_prediction_hash.json'))['sha256'])
            old_metrics=dict(np.load(halt_run/(tag+'_metrics.npz')))
            assert all(np.array_equal(metrics[k],v,equal_nan=True) for k,v in old_metrics.items())
            assert paths==read(halt_run/(tag+'_paths.json')) and groups==record['groups']
            val,_,_=evaluate(solver,ds,ids['val'],16,True,config['eval_batch'],halt=halt)
            assert val==record['validation']
            measured=representative_latency(solver,halt,ds,ids['val'])
            record['initial_first_problem_latency']=record['latency']
            record['initial_first_problem_baseline_latency']=record['baseline_latency']
            record['representative_paired_latency']=measured
            record['latency']=[{'batch':r['batch'],**r['adaptive'],'peak_increment_bytes':r['peak_increment_bytes_for_pair']} for r in measured]
            record['baseline_latency']=[{'batch':r['batch'],**r['fixed16'],'peak_increment_bytes':r['peak_increment_bytes_for_pair']} for r in measured]
            record['adoptable']=bool(maintains(val,record['baseline_validation']) and record['baseline_validation']['val']['success']>0
                and val['val']['steps']<16 and measured[0]['adaptive']['mean_ms']<measured[0]['fixed16']['mean_ms'])
            for group,row in groups.items():
                if not row['n']: continue
                hist=np.asarray(row['step_histogram']); mean_flops=sum(int(n)*(estimated_flops(t,True)+t*2*(128*64+64)) for t,n in enumerate(hist,1))/row['n']
                cost_rows.append({'seed':seed,'penalty':record['penalty'],'group':group,'n':row['n'],
                                  'mean_core_iterations':row['steps'],'estimated_mean_forward_flops_including_halt':float(mean_flops),
                                  'fixed16_solver_forward_flops':estimated_flops(16,True),
                                  'theoretical_flops_ratio':float(mean_flops/estimated_flops(16,True)),
                                  'fixed_encoder_and_readout_included':True})
            assert model_hash(solver)==frozen and all(p.grad is None for p in solver.parameters())
            shutil.copyfile(head_file,checkpoint_path(run,tag+'_best'))
            write(run/(tag+'_result.json'),record)
            replays[tag]={'full_test_logits_hash_exact':True,'paths_metrics_exact':True,'validation_exact':True,'weights_exact':True,
                          'exactly_one_halt_forward_per_core_forward':True}
            candidates.append(record)
            print('RUNTIME_REVIEW '+tag+' adoptable='+str(record['adoptable']),flush=True)
        feasible=[r for r in candidates if r['adoptable']]
        chosen=min(feasible,key=lambda r:(r['validation']['val']['steps'],r['latency'][0]['mean_ms'])) if feasible else None
        selection={**seed_group['selection'],'adopted_penalty':chosen['penalty'] if chosen else None,'fallback':None if chosen else 'fixed16',
                   'latency_basis':'stratified varied validation cases, paired measurement; no test selection'}
        revised.append({'selection':selection,'candidates':candidates}); write(run/'summary.json',revised)
    assert preservation()==read(run/'preservation_before.json')
    write(run/'verification.json',{'preservation_exact':True,'frozen_solver_exact':True,'all_candidates_restored':True,
                                 'replays':replays,'representative_latency_validation_only':True})
    write(run/'compute_summary.json',cost_rows)
    runtime_state={**state,'halt_training':str(halt_run),'halt':str(run),'runtime_review':str(run)}
    render(solver_run,run,destination=run/'figures')
    report(solver_run,run,runtime_state,figures_source=run/'figures')
    result=read(BASE/'SUMMARY.json'); result['runtime_review']=read(run/'review_provenance.json'); result['adaptive_compute_summary']=cost_rows
    result['initial_halting_measurements_preserved_at']=str(halt_run)
    latent=[]
    for seed in config['seeds']:
        d=dict(np.load(solver_run/f'B_seed{seed}_latent_diagnostics.npz'))
        latent.append({'seed':seed,'problem_ids':d['ids'].tolist(),'state_rms_change_mean':d['change'].mean(0).tolist(),
                       'input_attention_mean':d['input_attention'].mean(0).tolist(),'history_attention_mean':d['history_attention'].mean(0).tolist(),
                       'two_step_change_mean':d['two_step_change'].mean(0).tolist()})
    result['latent_diagnostic_summary']=latent; write(BASE/'SUMMARY.json',result)
    write(run/'complete.json',{'phase':'runtime_review','no_training':True,'policies':len(replays),'all_predictions_unchanged':True})
    print('RUNTIME_REVIEW_COMPLETE '+str(run),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('workflow',type=Path); args=parser.parse_args()
    setup(); review(args.workflow)
