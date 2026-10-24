"""Raw source -> labels -> model input -> route -> plotted coordinates cross checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import Core,Head,Baseline,features,loss,state_hash,parameters,flops
from download_benchmark import DATA,MAPS
from validate_benchmark import verify_raw
from parse_benchmark import read_official,parse_map
from prepare_dataset import load_dataset,SPLITS
from shortest_path import bfs,dijkstra
from evaluate import decode,legal_path,histories,predict,VARIANTS,evaluate_predictions

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[4]


def preservation():
    roots=[ROOT/'H001_dynamic_network',ROOT/'data']+[p for p in BASE.parent.iterdir() if p.is_dir() and p.name.startswith(('EXP001_','EXP002_','EXP003_'))]
    result={}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_relative_to(DATA): continue
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
            result[str(path.relative_to(ROOT))]=digest.hexdigest()
    return result


def structural():
    torch.manual_seed(42)
    core=Core(); head=Head(); p=torch.zeros(2,3072)
    p[:,1024+33]=1; p[:,2048+55]=1
    identities=tuple(id(x) for x in core.parameters()); inputs=[]; calls=[]
    def hook(module,args,output):
        inputs.append(args[0].detach().clone()); calls.append(tuple(id(x) for x in module.parameters()))
    handle=core.net.register_forward_hook(hook)
    states=core(p,8); handle.remove()
    assert len(calls)==8 and all(i==identities for i in calls)
    for k,x in enumerate(inputs):
        assert torch.equal(x[:,:3072],p)
        assert torch.equal(x[:,3072:],torch.ones_like(states[0]) if k==0 else states[k-1])
    assert all(x.shape==(2,1024) and torch.isfinite(x).all() for x in states)
    for x in states: x.retain_grad()
    loss(states[-1],torch.rand(2,1024),torch.ones(2,1024)).backward()
    gradients=[x.grad.abs().sum().item() for x in states]
    assert all(g>0 and np.isfinite(g) for g in gradients)
    before=state_hash(core)
    for x in core.parameters(): x.requires_grad_(False); x.grad=None
    h=histories(core,p)
    for mode in ('C','D','E'):
        x=features(p,h,mode); used=[]
        handler=head.register_forward_hook(lambda module,args,out:used.append(1))
        out=head(x); handler.remove()
        assert x.shape==(2,11264) and out.shape==(2,1024) and len(used)==1
        out.sum().backward()
    assert state_hash(core)==before and all(x.grad is None for x in core.parameters())
    assert flops(Baseline(3328))==8*flops(core)+flops(head)
    # G and S inside a .map are terrain, not markers.
    text='type octile\nheight 32\nwidth 32\nmap\n'+('\n'.join(['GS'+'.'*30]+['.'*32]*31))
    terrain,_=parse_map(text)
    assert not terrain[0,0] and not terrain[0,1]
    # Oracle and failure behavior of the common score-only decoder.
    walls=np.zeros(1024,dtype=np.uint8)
    field=bfs(walls,31*32+31)
    route=decode(field.astype(float),walls,0,1023)
    assert route['status']=='success' and len(route['path'])-1==62 and legal_path(route['path'],walls,0,1023)
    assert decode(np.ones(1024),walls,0,1023)['status']=='local_minimum'
    assert decode(field.astype(float),walls,0,1023,move_limit=1)['status']=='move_limit'
    # Decoder code cannot call a graph search or access a reference distance field.
    import ast,inspect
    tree=ast.parse(inspect.getsource(decode))
    assert not any(isinstance(n,ast.Name) and n.id.lower() in {'bfs','dijkstra','astar','a_star','target','optimal'} for n in ast.walk(tree))
    assert parameters(core)==1312000 and parameters(head)==3147008
    return {'core_parameters':parameters(core),'head_parameters':parameters(head),'A_large_parameters':parameters(Baseline(1088)),
            'A_flops_parameters':parameters(Baseline(3328)),'matched_flops':flops(Baseline(3328)),
            'all_state_gradient_l1':gradients,'shared_core':True,'context_W_S_G_preserved':True,
            'R0_all_ones':True,'no_teacher_forcing':True,'core_frozen':True,'single_head_call':True,
            'terrain_G_S_not_coordinates':True,'oracle_decoder_optimal':True,'no_search_in_decoder':True}


def verify_dataset(dataset_path,config):
    verify_raw()
    data=load_dataset(dataset_path)
    walls,raw_rows,_=read_official()
    assert np.array_equal(walls.reshape(5,1024),data['walls'])
    accepted=json.loads((dataset_path/'accepted_records.json').read_text(encoding='utf-8'))
    keys=set(); goal_splits={}; max_dijkstra_diff=0
    for field,(m,g) in zip(data['fields'],data['field_keys']):
        assert np.array_equal(field,bfs(walls[m],int(g)))
        other=dijkstra(walls[m],int(g))
        assert np.array_equal(field,other)
    oracle_success=0
    for i,row in enumerate(accepted):
        original=raw_rows[row['raw_row_index']]
        for key in original: assert row[key]==original[key]
        m,s,g=int(data['map_id'][i]),int(data['start'][i]),int(data['goal'][i])
        assert m==MAPS.index(row['map_name']) and s==row['sy']*32+row['sx'] and g==row['gy']*32+row['gx']
        assert (m,s,g) not in keys; keys.add((m,s,g))
        assert not walls[m].flat[s] and not walls[m].flat[g] and s!=g
        field=data['fields'][data['field_id'][i]]
        assert int(field[s])==data['optimal'][i]>=12
        goal_splits.setdefault((m,g),set()).add(int(data['split'][i]))
        oracle=decode(np.maximum(field,0).astype(float),walls[m],s,g,config['decoder_move_limit'])
        assert oracle['status']=='success' and len(oracle['path'])-1==data['optimal'][i]
        assert legal_path(oracle['path'],walls[m],s,g)
        oracle_success+=1
    assert all(len(v)==1 for v in goal_splits.values())
    heldout={MAPS.index(n) for n in config['heldout_maps']}
    assert set(data['map_id'][data['split']==3])==heldout
    assert not (set(data['map_id'][data['split']<3])&heldout)
    edges=np.quantile(data['optimal'][data['split']==0],config['difficulty_quantiles_from_training_only'],method='nearest').astype(int)
    assert np.array_equal(edges,config['adopted_distance_edges'])
    assert np.array_equal(np.searchsorted(edges,data['optimal'],side='left'),data['band'])
    return {'official_accepted_rows':len(accepted),'independent_goal_fields':len(data['fields']),
            'oracle_success_and_optimal_rate':oracle_success/len(accepted),'no_SG_or_goal_field_split_overlap':True,
            'heldout_maps_absent_from_training':True,'official_coordinates_match_all_inputs':True,
            'distance_quartiles_training_only':edges.tolist()}


def verify_run(run,config,device,original):
    from train import make_tensors
    reference=json.loads((run/'dataset_reference.json').read_text(encoding='utf-8'))
    dataset_path=Path(reference['path'])
    for name,digest in reference['files'].items(): assert hashlib.sha256((dataset_path/name).read_bytes()).hexdigest()==digest
    result={'structure':structural(),'dataset':verify_dataset(dataset_path,config),'replay':{}}
    data=load_dataset(dataset_path); ids=dict(np.load(run/'used_indices.npz'))
    tensor={s:make_tensors(data,ids[s],device,config['distance_scale']) for s in ('train','val')}
    tensor['evaluation']=make_tensors(data,ids['evaluation'],device,config['distance_scale'])
    checkpoints=BASE/'checkpoints'/run.name
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    for seed in config['seeds']:
        saved=torch.load(checkpoints/f'core_seed{seed}.pt',map_location=device,weights_only=False)
        core=Core().to(device); core.load_state_dict(saved['model'])
        assert state_hash(core)==saved['final_hash']
        for p in core.parameters(): p.requires_grad_(False); p.grad=None
        hashes=json.loads((run/f'cache_seed{seed}.json').read_text(encoding='utf-8'))['sha256']; cache={}
        for split in ('train','val','evaluation'):
            h=histories(core,tensor[split]['problem'],config['eval_batch_size'])
            assert hashlib.sha256(h.cpu().numpy().tobytes()).hexdigest()==hashes[split]
            if split=='evaluation': cache=h
        assert np.array_equal(cache.cpu().numpy(),np.load(run/f'core_seed{seed}_evaluation_history.npy',mmap_mode='r'))
        initial_hashes=set(); orders=set()
        for record in [r for r in summary if r['seed']==seed]:
            mode=record['mode']
            if mode=='B': model=None
            else:
                saved_head=torch.load(checkpoints/(record['name']+'.pt'),map_location=device,weights_only=False)
                model=Head() if mode in ('C','D','E') else Baseline(config['baseline_widths'][mode])
                model=model.to(device); model.load_state_dict(saved_head['model'])
                assert saved_head['core_hash']==state_hash(core)
                if mode in ('C','D','E'): initial_hashes.add(saved_head['initial_hash']); orders.add(saved_head['order_hash'])
            out=predict(model,tensor['evaluation']['problem'],cache if mode in ('B','C','D') else None,mode,config['eval_batch_size'])
            assert np.array_equal(out,np.load(run/f'{mode}_seed{seed}_scores.npy',mmap_mode='r'))
            values,paths=evaluate_predictions(out,data,ids['evaluation'],config)
            stored=dict(np.load(run/f'{mode}_seed{seed}_sample_metrics.npz'))
            assert all(np.array_equal(values[k],stored[k],equal_nan=True) for k in values)
            assert paths==json.loads((run/f'{mode}_seed{seed}_paths.json').read_text(encoding='utf-8'))
            if mode=='D':
                expected=json.loads((run/f'D_seed{seed}_ablation_prediction_hashes.json').read_text(encoding='utf-8'))
                for variant in VARIANTS:
                    out_variant=predict(model,tensor['evaluation']['problem'],cache,'D',config['eval_batch_size'],variant)
                    assert hashlib.sha256(out_variant.tobytes()).hexdigest()==expected[variant]
            result['replay'][record['name']]={'max_score_difference':0.,'routes_and_metrics_exact':True}
        assert len(initial_hashes)==1 and len(orders)==1
        torch.manual_seed(seed+50000); assert state_hash(Head())==next(iter(initial_hashes))
        assert state_hash(core)==saved['final_hash'] and all(p.grad is None for p in core.parameters())
    assert len(result['replay'])==len(config['seeds'])*7
    source_hashes=json.loads((run/'source_hashes.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((run/'source'/n).read_bytes()).hexdigest()==h for n,h in source_hashes.items())
    for snapshot in run.glob('resume_source_*'):
        expected=json.loads((snapshot/'hashes.json').read_text(encoding='utf-8'))
        assert all(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()==h for n,h in expected.items())
    assert preservation()==original
    result.update(preserved_files=len(original),preservation_exact=True,source_snapshot_exact=True,
                  CDE_initializations_and_orders_match=True,frozen_core_unchanged=True,
                  all_ablation_predictions_reproduced=True)
    return result


def verify_visualization(run):
    info=json.loads((run/'visualization_payload.json').read_text(encoding='utf-8'))
    reference=json.loads((run/'dataset_reference.json').read_text(encoding='utf-8'))
    data=load_dataset(reference['path']); raw_walls,raw_rows,_=read_official()
    for sample in info['samples']:
        index=sample['record_index']; m=int(data['map_id'][index])
        assert sample['wall_sha256']==hashlib.sha256(raw_walls[m].reshape(1024).tobytes()).hexdigest()
        assert sample['start_xy']==[int(data['start'][index]%32),int(data['start'][index]//32)]
        assert sample['goal_xy']==[int(data['goal'][index]%32),int(data['goal'][index]//32)]
        assert legal_path(sample['oracle_path'],raw_walls[m],int(data['start'][index]),int(data['goal'][index]))
        assert len(sample['oracle_path'])-1==data['optimal'][index]
        for filename in sample['images']: assert (run/filename).is_file()
    return {'samples':len(info['samples']),'official_walls_SG_and_oracle_paths_match':True,
            'coordinate_axes':'imshow origin=upper; scatter x=official x, y=official y; lines use same coordinates'}


if __name__=='__main__':
    import argparse,os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path); args=parser.parse_args()
    run=args.run.resolve(); config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
    result=verify_run(run,config,torch.device(config['device']),json.loads((run/'preservation_before.json').read_text(encoding='utf-8')))
    if (run/'visualization_payload.json').exists(): result['visualization']=verify_visualization(run)
    (run/'verification_recheck.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verification passed',flush=True)
