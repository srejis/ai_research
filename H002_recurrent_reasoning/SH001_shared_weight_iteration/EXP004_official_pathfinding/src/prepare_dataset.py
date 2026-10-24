"""Independent data audit command; no synthetic scenarios or silently relaxed filters."""
from pathlib import Path
import hashlib
import json
import numpy as np
from download_benchmark import DATA,MAPS,download
from validate_benchmark import validate

BASE=Path(__file__).resolve().parents[1]
SPLITS=['train','val','test_seen','test_unseen']


def dataset_signature(config,manifest):
    selected={key:config[key] for key in ('maps','heldout_maps','split_seed','seen_goal_split_fractions','minimum_four_way_distance','difficulty_quantiles_from_training_only')}
    selected['archives']={r['archive']:r['sha256'] for r in manifest['files']}
    return hashlib.sha256(json.dumps(selected,sort_keys=True).encode('utf-8')).hexdigest()[:16]


def prepare(config):
    manifest=download()
    key=dataset_signature(config,manifest)
    out=DATA/'prepared'/key
    if (out/'complete.json').exists():
        info=json.loads((out/'complete.json').read_text(encoding='utf-8'))
        if hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()!=info['dataset_sha256']:
            raise RuntimeError('Prepared dataset hash mismatch')
        return out
    if out.exists():
        raise RuntimeError(f'Incomplete preparation retained at {out}; inspect it explicitly before a new preparation')
    walls,fields,field_keys,accepted,validation=validate(config,out)
    rng=np.random.default_rng(config['split_seed'])
    splits=np.full(len(accepted),-1,dtype=np.int8)
    m=np.array([r['map_id'] for r in accepted],dtype=np.int16)
    goals=np.array([r['goal'] for r in accepted],dtype=np.int16)
    for map_id,name in enumerate(MAPS):
        mask=m==map_id
        if name in config['heldout_maps']:
            splits[mask]=3
            continue
        unique_goals=np.unique(goals[mask])
        rng.shuffle(unique_goals)
        fractions=config['seen_goal_split_fractions']
        if len(fractions)!=3 or not np.isclose(sum(fractions),1):
            raise ValueError('Expected three split fractions summing to one')
        train_end=int(len(unique_goals)*fractions[0]); val_end=int(len(unique_goals)*(fractions[0]+fractions[1]))
        for index,part in enumerate((unique_goals[:train_end],unique_goals[train_end:val_end],unique_goals[val_end:])):
            splits[mask&np.isin(goals,part)]=index
    assert np.all(splits>=0) and all((splits==i).any() for i in range(4))
    optimal=np.array([r['optimal_four_way'] for r in accepted],dtype=np.int16)
    edges=np.quantile(optimal[splits==0],config['difficulty_quantiles_from_training_only'],method='nearest').astype(int)
    if len(np.unique(edges))!=3:
        raise RuntimeError('Training quantiles do not create four distinct bands; no silent band rewrite')
    band=np.searchsorted(edges,optimal,side='left').astype(np.int8)
    array={'walls':walls.reshape(5,1024),'fields':fields,'field_keys':field_keys,
           'map_id':m,'start':np.array([r['start'] for r in accepted],dtype=np.int16),
           'goal':goals,'field_id':np.array([r['field'] for r in accepted],dtype=np.int32),
           'split':splits,'optimal':optimal,'band':band,
           'original_cost':np.array([r['original_cost'] for r in accepted]),
           'manhattan':np.array([r['manhattan'] for r in accepted],dtype=np.int16),
           'detour':np.array([r['detour_ratio'] for r in accepted],dtype=np.float32),
           'family':np.array([r['scenario_type']=='random' for r in accepted],dtype=np.int8)}
    np.savez_compressed(out/'dataset.npz',**array)
    (out/'accepted_records.json').write_text(json.dumps(accepted,indent=2),encoding='utf-8')
    audit={'dataset_id':key,'difficulty_edges_four_way':edges.tolist(),
           'difficulty_rule':'training-only nearest quartiles, right-closed bins; labels Q1/Q2/Q3/Q4',
           'minimum_distance':12,'split_unit':'map+goal to avoid identical full-field target leakage',
           'heldout_maps':config['heldout_maps'],'splits':{},'validation':validation}
    for i,name in enumerate(SPLITS):
        mask=splits==i
        audit['splits'][name]={'n':int(mask.sum()),'map_counts':{n:int((mask&(m==j)).sum()) for j,n in enumerate(MAPS)},
                               'distance_min':int(optimal[mask].min()),'distance_mean':float(optimal[mask].mean()),
                               'distance_max':int(optimal[mask].max()),
                               'bands':{f'Q{k+1}':int((mask&(band==k)).sum()) for k in range(4)},
                               'unique_map_goals':len({(int(a),int(b)) for a,b in zip(m[mask],goals[mask])})}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    completion={'dataset_sha256':hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest(),'dataset_id':key}
    (out/'complete.json').write_text(json.dumps(completion,indent=2),encoding='utf-8')
    return out


def load_dataset(path):
    return dict(np.load(Path(path)/'dataset.npz'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=BASE/'configs/experiment.json')
    args=parser.parse_args()
    out=prepare(json.loads(args.config.read_text(encoding='utf-8')))
    print('DATASET='+str(out),flush=True)
    print(json.dumps(json.loads((out/'audit.json').read_text(encoding='utf-8'))['splits'],indent=2))
