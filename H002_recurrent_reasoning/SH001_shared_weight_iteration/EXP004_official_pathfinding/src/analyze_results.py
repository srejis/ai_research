"""Posthoc evaluation of saved routes only; never changes models or decoding."""
import argparse
import math
import hashlib
import json
from pathlib import Path
import numpy as np

MODES=('A','A_large','A_flops','B','C','D','E')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(run):
    assert read(run/'status.json')['status']=='complete'
    config=read(run/'config.json')
    data=dict(np.load(Path(read(run/'dataset_reference.json')['path'])/'dataset.npz'))
    ids=np.load(run/'used_indices.npz')['evaluation']
    edges=np.quantile(data['manhattan'][data['split']==0],[.25,.5,.75],method='nearest')
    manhattan=np.searchsorted(edges,data['manhattan'][ids],side='left')
    detour=np.searchsorted([1.,1.5,2.],data['detour'][ids],side='left')
    # Canonical shortest route: first exact one-step distance decrease in U/D/L/R order.
    # This is evaluation metadata, not a model route or a minimum-turn oracle.
    turns=[]
    for i in ids:
        field=data['fields'][data['field_id'][i]]
        at=int(data['start'][i]); goal=int(data['goal'][i]); directions=[]
        while at!=goal:
            y,x=divmod(at,32)
            candidates=[(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
            for direction,(yy,xx) in enumerate(candidates):
                if 0<=yy<32 and 0<=xx<32 and field[yy*32+xx]==field[at]-1:
                    at=yy*32+xx; directions.append(direction); break
            else: raise AssertionError('No oracle descent')
        assert len(directions)==data['optimal'][i]
        turns.append(sum(a!=b for a,b in zip(directions,directions[1:])))
    turn_band=np.searchsorted([2,5,10],turns,side='left')
    np.savez_compressed(run/'additional_difficulty_metadata.npz',record_indices=ids,canonical_turns=turns,
                        manhattan_band=manhattan,detour_band=detour,turn_band=turn_band)
    groups={}
    for split,label in ((2,'seen'),(3,'unseen')):
        base=data['split'][ids]==split
        groups[label]=base
        for attribute,bands in [('manhattan',manhattan),('detour',detour),('canonical_turns',turn_band)]:
            for k in range(4): groups[f'{label}/{attribute}/bin{k+1}']=base&(bands==k)
    records=[]; failures=[]; paired=[]
    for seed in config['seeds']:
        values={m:dict(np.load(run/f'{m}_seed{seed}_sample_metrics.npz')) for m in MODES}
        for mode in MODES:
            v=values[mode]
            for group,mask in groups.items():
                records.append({'seed':seed,'mode':mode,'group':group,'n':int(mask.sum()),
                                **{key:float(v[key][mask].mean()) if mask.any() else None
                                   for key in ('success','spl','distance_mae','optimal')}})
            routes=read(run/f'{mode}_seed{seed}_paths.json')
            remaining=np.array([data['fields'][data['field_id'][i]][r['path'][-1]] for i,r in zip(ids,routes)])
            assert all(r['record_index']==int(i) for i,r in zip(ids,routes))
            for split,label in ((2,'seen'),(3,'unseen')):
                mask=(data['split'][ids]==split)&(v['success']==0)
                failures.append({'seed':seed,'mode':mode,'split':label,'failed':int(mask.sum()),
                                 'mean_true_steps_remaining':float(remaining[mask].mean()) if mask.any() else None,
                                 'failed_within_four_steps_of_goal':int((mask&(remaining<=4)).sum()),
                                 'failed_at_start':int((mask&(v['path_length']==0)).sum())})
                mask=data['split'][ids]==split
                c=values['C']['success']>0; d=values['D']['success']>0
                if mode=='D': paired.append({'seed':seed,'split':label,'n':int(mask.sum()),
                    'D_only':int((mask&d&~c).sum()),'C_only':int((mask&c&~d).sum()),
                    'both_success':int((mask&c&d).sum()),'both_failure':int((mask&~c&~d).sum())})
    aggregate=[]
    for mode in MODES:
        for group in groups:
            rows=[r for r in records if r['mode']==mode and r['group']==group]
            result={'mode':mode,'group':group,'n':rows[0]['n']}
            for key in ('success','spl','distance_mae','optimal'):
                vals=[r[key] for r in rows if r[key] is not None]
                result[key+'_mean']=float(np.mean(vals)) if vals else None
                result[key+'_std']=float(np.std(vals,ddof=1 if len(vals)>1 else 0)) if vals else None
            aggregate.append(result)
    indices=dict(np.load(run/'used_indices.npz'))
    batches={k:math.ceil(len(indices[k])/config['eval_batch_size']) for k in ('train','val','evaluation')}
    train_batches=math.ceil(len(indices['train'])/config['batch_size'])
    calls={'unit':'one forward of a module on a batch; optimizer/backward not counted',
           'scope':'per seed; core trained once and cache shared across B/C/D',
           'core_training':8*train_batches*config['epochs_core'],
           'core_validation':8*batches['val']*config['epochs_core'],
           'core_cache_train_val_test':8*sum(batches.values()),
           'each_head_training':train_batches*config['epochs_heads'],
           'each_head_validation':batches['val']*config['epochs_heads'],
           'each_baseline_training':train_batches*config['epochs_baselines'],
           'each_baseline_validation':batches['val']*config['epochs_baselines'],
           'each_head_or_baseline_normal_evaluation':batches['evaluation'],
           'D_perturbation_head_evaluation':13*batches['evaluation'],
           'latency_whole_system_forwards_each_model':config['latency_warmup']+3*config['latency_repeats'],
           'caveat':'Structural checks, verification replay, and interrupted/resumed cache recomputation are additional verification overhead, excluded from scientific training budget.'}
    output={'definitions':{'manhattan_training_quartiles':edges.astype(int).tolist(),
            'detour_bins':['<=1','(1,1.5]','(1.5,2]','>2'],
            'canonical_turn_bins':['0-2','3-5','6-10','>10'],
            'canonical_turn_caveat':'U/D/L/R tie-break shortest route; not minimum turns; evaluation only',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'groups':aggregate,'seed_groups':records,'failures':failures,'paired_C_D':paired,'module_forward_calls':calls}
    (run/'additional_analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print('Additional route analysis complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path)
    analyze(parser.parse_args().run.resolve())
