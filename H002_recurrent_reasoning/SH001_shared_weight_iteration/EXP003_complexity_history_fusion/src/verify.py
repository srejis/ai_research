"""Independent label verification, frozen-core and single-use readout contracts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import Core,Readout,Baseline,features,state_hash,masked_loss,parameters,linear_flops
from maze_generator import generate,audit,local_map,bfs
from evaluate import cache_history,predict,sample_metrics,INTERVENTIONS

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[4]


def preservation():
    roots=[ROOT/'H001_dynamic_network',ROOT/'data',BASE.parent/'EXP001_recurrent_mlp',BASE.parent/'EXP002_maze_propagation']
    result={}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(chunk)
                result[str(path.relative_to(ROOT))]=digest.hexdigest()
    return result


def flood_distances(problem):
    valid=problem[:256].reshape(16,16).astype(bool)
    walls=problem[256:512].reshape(16,16).astype(bool)
    found=problem[512:].reshape(16,16).astype(bool)
    distances=np.full((16,16),-1,dtype=np.int16)
    distances[found]=0
    for k in range(1,257):
        adjacent=np.zeros_like(found)
        adjacent[1:]|=found[:-1]
        adjacent[:-1]|=found[1:]
        adjacent[:,1:]|=found[:,:-1]
        adjacent[:,:-1]|=found[:,1:]
        new=adjacent&valid&~walls&~found
        if not new.any():
            break
        distances[new]=k
        found|=new
    return distances.reshape(256)


def structural():
    torch.manual_seed(42)
    core,head=Core(),Readout()
    assert parameters(core)==328192 and parameters(head)==786944
    assert abs(parameters(Baseline(1088))-(parameters(core)+parameters(head)))/(parameters(core)+parameters(head))<.001
    assert linear_flops(Baseline(3328))==8*linear_flops(core)+linear_flops(head)
    problem=torch.zeros(3,768)
    problem[:,:256]=1
    problem[0,:128]=0
    problem[:,512+200]=1
    core_ids=tuple(id(p) for p in core.parameters())
    calls,inputs=[],[]
    def hook(module,args,out):
        calls.append(tuple(id(p) for p in module.parameters()))
        inputs.append(args[0].detach().clone())
    handle=core.net.register_forward_hook(hook)
    states=core(problem,8)
    handle.remove()
    assert len(calls)==8 and all(i==core_ids for i in calls)
    for i,x in enumerate(inputs):
        assert torch.equal(x[:,:768],problem)
        assert torch.equal(x[:,768:],problem[:,512:] if i==0 else states[i-1])
    assert all(x.shape==(3,256) and torch.isfinite(x).all() for x in states)
    assert all((x[0,:128]==0).all() for x in states)
    for x in states:
        x.retain_grad()
    masked_loss(states[-1],torch.rand(3,256),problem[:,:256]).backward()
    gradients=[x.grad.abs().sum().item() for x in states]
    assert all(g>0 and np.isfinite(g) for g in gradients)
    history=torch.stack([x.detach() for x in states],dim=1)
    before=state_hash(core)
    for p in core.parameters():
        p.requires_grad_(False)
        p.grad=None
    frozen_outputs=cache_history(core,problem)
    assert torch.equal(history,frozen_outputs)
    initial=state_hash(head)
    outputs=[]
    for mode in ('C','D','E'):
        x=features(problem,frozen_outputs,mode)
        assert x.shape==(3,2816)
        calls_head=[]
        h=head.register_forward_hook(lambda module,args,out:calls_head.append(1))
        output=head(x)
        h.remove()
        assert output.shape==(3,256) and len(calls_head)==1
        if mode=='C':
            assert torch.equal(x[:,768:].reshape(3,8,256),frozen_outputs[:,-1:].expand(-1,8,-1))
        if mode=='D':
            assert torch.equal(x[:,768:].reshape(3,8,256),frozen_outputs)
        if mode=='E':
            assert (x[:,768:]==0).all()
        outputs.append(output)
    sum(x.sum() for x in outputs).backward()
    assert state_hash(core)==before and all(p.grad is None for p in core.parameters())
    assert state_hash(head)==initial
    hand=np.ones((4,4),dtype=np.uint8)
    hand.flat[[0,1,5,15]]=0
    assert bfs(hand,0,4)[[0,1,5,15]].tolist()==[0,1,2,-1]
    return {'core_parameters':parameters(core),'head_parameters':parameters(head),
            'large_A_parameters':parameters(Baseline(1088)),'core_gradient_l1':gradients,
            'shared_core_calls':len(calls),'context_preserved':True,'no_teacher_forcing':True,
            'padding_masked':True,'one_readout_call_no_feedback':True,'frozen_core_no_grad':True,
            'large_A_parameter_relative_gap':(parameters(Baseline(1088))-parameters(core)-parameters(head))/(parameters(core)+parameters(head)),
            'flops_matched_A':linear_flops(Baseline(3328)),'manual_bfs':True}


def verify_run(run,config,device,original):
    data=dict(np.load(run/'dataset.npz'))
    regenerated,_=generate(config)
    assert all(np.array_equal(data[k],regenerated[k]) for k in data)
    result={'structure':structural(),'dataset':audit(data,config),'checkpoints':{},'history_cache':{}}
    count=0
    for split in ('train','val','test'):
        for problem,dist in zip(data[split+'_problem'],data[split+'_distances']):
            assert np.array_equal(flood_distances(problem),dist)
            count+=1
    result['independent_flood_cases']=count
    checkpoints=BASE/'checkpoints'/run.name
    problems={s:torch.from_numpy(data[s+'_problem']).to(device) for s in ('train','val','test')}
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    for seed in config['seeds']:
        saved=torch.load(checkpoints/f'core_seed{seed}.pt',map_location=device,weights_only=False)
        core=Core().to(device)
        core.load_state_dict(saved['model'])
        assert state_hash(core)==saved['final_sha256']
        for p in core.parameters():
            p.requires_grad_(False)
            p.grad=None
        history_test=None
        for split in ('train','val','test'):
            cached=cache_history(core,problems[split],256)
            expected=np.load(run/f'history_seed{seed}_{split}.npz')['states']
            assert np.array_equal(cached.cpu().numpy(),expected)
            if split=='test':
                history_test=cached
        result['history_cache'][str(seed)]='all splits exactly equal to frozen-core output; labels not used'
        initial_hashes,orders=set(),set()
        for s in [r for r in summary if r['seed']==seed]:
            mode=s['mode']
            if mode=='B':
                output,logits=predict(None,problems['test'],history_test,'B')
            else:
                saved_head=torch.load(checkpoints/(s['name']+'.pt'),map_location=device,weights_only=False)
                model=Readout() if mode in ('C','D','E') else Baseline(config['baseline_widths'][mode])
                model=model.to(device)
                model.load_state_dict(saved_head['model'])
                assert saved_head['core_sha256']==state_hash(core)
                assert saved_head['core_unchanged_after_training']
                if mode in ('C','D','E'):
                    initial_hashes.add(saved_head['initial_sha256'])
                    orders.add(saved_head['batch_order_sha256'])
                output,logits=predict(model,problems['test'],history_test if mode in ('C','D') else None,mode)
                expected_steps=(config['epochs_heads'] if mode in ('C','D','E') else config['epochs_baselines'])*((len(problems['train'])+config['batch_size']-1)//config['batch_size'])
                assert saved_head['optimizer_steps']==expected_steps
                if mode=='D':
                    for variant in INTERVENTIONS:
                        p,z=predict(model,problems['test'],history_test,'D',perturbation=variant)
                        original_prediction=np.load(run/f'D_seed{seed}_ablation_{variant}.npz')
                        assert np.array_equal(p,original_prediction['probability']) and np.array_equal(z,original_prediction['logits'])
            recorded=np.load(run/(s['name']+'_prediction.npz'))
            assert np.array_equal(output,recorded['probability'])
            if logits is not None:
                assert np.array_equal(logits,recorded['logits'])
            assert np.isfinite(output).all()
            # Independent torch masked stable BCE cross-checks NumPy metrics in FP64.
            target=torch.from_numpy((data['test_distances']>=0).astype(np.float64))
            valid=torch.from_numpy(data['test_problem'][:,:256].astype(np.float64))
            pred=torch.from_numpy(logits if logits is not None else output).double()
            bce=masked_loss(pred,target,valid,logits=logits is not None).item()
            values=sample_metrics(output,data['test_problem'],data['test_distances'],data['test_start'],logits)
            assert abs(bce-values['bce'].mean())<1e-10
            pred_bool=(torch.from_numpy(output)>=.5)&valid.bool()
            truth=target.bool()
            iou=((pred_bool&truth).sum(1).double()/(pred_bool|truth).sum(1).clamp_min(1)).mean().item()
            assert abs(iou-values['iou'].mean())<1e-12
            result['checkpoints'][s['name']]={'max_output_difference':0.,'independent_bce_difference':abs(bce-values['bce'].mean())}
        assert len(initial_hashes)==1 and len(orders)==1
        torch.manual_seed(seed+50000)
        assert state_hash(Readout())==next(iter(initial_hashes))
        assert all(p.grad is None for p in core.parameters()) and state_hash(core)==saved['final_sha256']
    assert len(result['checkpoints'])==7*len(config['seeds'])
    source_hashes=json.loads((run/'source_sha256.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest for name,digest in source_hashes.items())
    assert preservation()==original,'Existing research modified'
    result.update(preserved_files=len(original),preservation_exact=True,source_snapshot_hashes_match=True,
                  paired_head_initializations_and_batches=True,core_frozen_after_all_heads=True,
                  all_predictions_and_ablations_reproduced=True)
    return result


if __name__=='__main__':
    import argparse
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run.resolve()
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    result=verify_run(run,config,torch.device(config['device']),json.loads((run/'preservation_before.json').read_text(encoding='utf-8')))
    (run/'verification_recheck.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verification passed',flush=True)
