"""Size and BFS distance are separate axes; empty strata remain explicit."""
import csv
import json
from pathlib import Path
import numpy as np
from evaluate import sample_metrics,INTERVENTIONS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def groups(meta,config):
    masks={'all':np.ones(len(meta),dtype=bool)}
    for size in config['maze_sizes']:
        masks[f'size/{size}']=meta[:,0]==size
    for label,(lo,hi) in config['difficulty_bands'].items():
        mask=(meta[:,1]>=lo)&(meta[:,1]<=hi)
        masks[f'difficulty/{label}']=mask
        for size in config['maze_sizes']:
            masks[f'size/{size}/difficulty/{label}']=mask&(meta[:,0]==size)
    for column,label,limits in [(2,'branches',(4,16)),(3,'dead_ends',(4,16)),(4,'components',(2,5))]:
        a,b=limits
        masks[f'{label}/le{a}']=meta[:,column]<=a
        masks[f'{label}/{a+1}to{b}']=(meta[:,column]>a)&(meta[:,column]<=b)
        masks[f'{label}/gt{b}']=meta[:,column]>b
    masks['family/random_obstacles']=meta[:,9]==0
    masks['family/cut_corridors']=meta[:,9]==1
    masks['detour/le1.5']=meta[:,7]<=1.5
    masks['detour/1.5to3']=(meta[:,7]>1.5)&(meta[:,7]<=3)
    masks['detour/gt3']=meta[:,7]>3
    return masks


def reduce_samples(values,masks,mode,seed,variant='normal'):
    rows=[]
    for group,mask in masks.items():
        row={'mode':mode,'seed':seed,'variant':variant,'group':group,'n':int(mask.sum())}
        if mask.any():
            row.update({key:float(value[mask].mean()) for key,value in values.items()})
        rows.append(row)
    return rows


def aggregate(rows):
    result=[]
    keys=('iou','f1','bce','exact','start_accuracy','fnr','disconnected_fpr','confident_wrong','pixel_error')
    identities=sorted({(r['mode'],r['variant'],r['group']) for r in rows})
    for mode,variant,group in identities:
        values=[r for r in rows if (r['mode'],r['variant'],r['group'])==(mode,variant,group)]
        row={'mode':mode,'variant':variant,'group':group,'n':values[0]['n'],'seeds':len(values)}
        if row['n']:
            for key in keys:
                vector=[r[key] for r in values]
                row[key+'_mean']=float(np.mean(vector))
                row[key+'_std']=float(np.std(vector,ddof=1 if len(vector)>1 else 0))
        result.append(row)
    return result


def analyze(run):
    run=Path(run)
    config=read(run/'config.json')
    summary=read(run/'summary.json')
    data=np.load(run/'dataset.npz')
    problem,distances,start,meta=[data['test_'+k] for k in ('problem','distances','start','meta')]
    masks=groups(meta,config)
    rows,ablations=[],[]
    for s in summary:
        with np.load(run/(s['name']+'_prediction.npz')) as file:
            values=sample_metrics(file['probability'],problem,distances,start,file['logits'] if 'logits' in file else None)
        assert all(np.isfinite(value).all() for value in values.values())
        np.savez_compressed(run/(s['name']+'_sample_metrics.npz'),**values)
        rows.extend(reduce_samples(values,masks,s['mode'],s['seed']))
        if s['mode']=='D':
            for variant in INTERVENTIONS:
                with np.load(run/f'D_seed{s["seed"]}_ablation_{variant}.npz') as file:
                    values=sample_metrics(file['probability'],problem,distances,start,file['logits'])
                ablations.extend(reduce_samples(values,masks,'D',s['seed'],variant))
    reduced,perturbed=aggregate(rows),aggregate(ablations)
    for filename,value in [('group_metrics.json',rows),('aggregate.json',reduced),('ablation_metrics.json',ablations),('ablation_aggregate.json',perturbed)]:
        (run/filename).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    with (run/'comparison.csv').open('w',encoding='utf-8',newline='') as file:
        fields=['mode','variant','group','n','seeds']+[k+suffix for k in ('iou','f1','bce','exact','start_accuracy','fnr','disconnected_fpr','confident_wrong','pixel_error') for suffix in ('_mean','_std')]
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(reduced)
    differences=[]
    for group in masks:
        for other in ('A','A_large','A_flops','B','C','E'):
            delta=[]
            for seed in config['seeds']:
                d=next(r for r in rows if r['mode']=='D' and r['group']==group and r['seed']==seed)
                baseline=next(r for r in rows if r['mode']==other and r['group']==group and r['seed']==seed)
                if d['n']:
                    delta.append(d['iou']-baseline['iou'])
            differences.append({'group':group,'comparison':'D-'+other,'n':int(masks[group].sum()),
                                'mean':float(np.mean(delta)) if delta else None,
                                'std':float(np.std(delta,ddof=1 if len(delta)>1 else 0)) if delta else None,
                                'positive_seeds':sum(x>0 for x in delta)})
    (run/'paired_gaps.json').write_text(json.dumps(differences,indent=2),encoding='utf-8')
    return reduced


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    analyze(parser.parse_args().run)
