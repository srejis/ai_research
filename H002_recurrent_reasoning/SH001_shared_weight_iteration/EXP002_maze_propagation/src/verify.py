"""Independent BFS oracles, recurrence/BPTT proof, replay and preservation audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import MazeMLP
from maze_generator import bfs,generate,audit
from evaluate import evaluate

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[1]


def preservation():
    roots = [ROOT/'H001_dynamic_network',ROOT/'data',BASE.parent/'EXP001_recurrent_mlp']
    result = {}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(chunk)
                result[str(path.relative_to(ROOT))] = digest.hexdigest()
    return result


def flood_oracle(walls,goal):
    """Synchronous four-neighbor boolean dilation, independent of BFS queue/indexing."""
    open_grid = ~walls.astype(bool).reshape(8,8)
    found = np.zeros((8,8),dtype=bool)
    found.flat[goal] = True
    states = [found.copy()]
    for _ in range(64):
        adjacent = np.zeros_like(found)
        adjacent[1:] |= found[:-1]
        adjacent[:-1] |= found[1:]
        adjacent[:,1:] |= found[:,:-1]
        adjacent[:,:-1] |= found[:,1:]
        updated = found | (adjacent & open_grid)
        states.append(updated.copy())
        if np.array_equal(updated,found):
            break
        found = updated
    return states


def bfs_checks():
    # Hand maze: goal (0,0), path right-right-down, plus disconnected open (7,7).
    walls = np.ones(64,dtype=np.uint8)
    walls[[0,1,2,10,63]] = 0
    distance = bfs(walls,0)
    assert distance[[0,1,2,10]].tolist() == [0,1,2,3]
    assert distance[63] == -1 and (distance>=0).sum() == 4
    diagonal = np.ones(64,dtype=np.uint8)
    diagonal[[0,9]] = 0
    assert bfs(diagonal,0)[9] == -1
    full = bfs(np.zeros(64,dtype=np.uint8),0).reshape(8,8)
    assert np.array_equal(full,np.indices((8,8)).sum(0))
    return {'hand_path_and_disconnected_region':True,'no_diagonal':True,'open_grid_manhattan_distances':True}


def structural():
    results = {}
    for mode in 'ABCD':
        torch.manual_seed(42)
        model = MazeMLP()
        assert sum(p.numel() for p in model.parameters()) == 98752
        walls,goal = torch.rand(2,64),torch.rand(2,64)
        ids = tuple(id(p) for p in model.parameters())
        calls,inputs = [],[]
        def hook(module,args,output):
            calls.append(tuple(id(p) for p in module.parameters()))
            inputs.append(args[0].detach().clone())
        handle = model.core.register_forward_hook(hook)
        states = model(walls,goal,8,mode)
        handle.remove()
        assert len(calls)==8 and all(c==ids for c in calls)
        assert all(h.shape==(2,192) for h in states)
        assert all(torch.isfinite(h).all() for h in states)
        if mode in 'CD':
            assert all(torch.equal(h[:,:64],walls) and torch.equal(h[:,64:128],goal) for h in inputs)
        else:
            assert all(torch.equal(inputs[i],states[i-1]) for i in range(1,8))
        for h in states:
            h.retain_grad()
        target = torch.rand(2,64)
        loss = torch.stack([F.binary_cross_entropy(h[:,128:],target) for h in states]).mean() if mode=='D' else F.binary_cross_entropy(states[-1][:,128:],target)
        loss.backward()
        gradient = [h.grad[:,128:].abs().sum().item() for h in states]
        assert all(np.isfinite(g) and g>0 for g in gradient)
        ignored_grad = model.core[2].weight.grad[:128].abs().sum().item()
        if mode in 'CD':
            assert ignored_grad == 0
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            model(walls,goal,32,mode)
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        results[mode] = {'nominal_parameters':98752,'shared_calls':len(calls),
                         'r_state_gradient_l1':gradient,'ignored_head_gradient_l1':ignored_grad,
                         'inference_weights_unchanged':True}
    return results


def verify_run(run,config,device,original):
    data = dict(np.load(run/'dataset.npz'))
    regenerated,_ = generate(config)
    assert all(np.array_equal(data[k],regenerated[k]) for k in data)
    result = {'structure':structural(),'bfs_hand_checks':bfs_checks(),'dataset':audit(data)}
    # Every stored label is checked by a separate non-queue implementation.
    oracle_count = 0
    for split in ('train','val','test'):
        for w,g,d in zip(data[split+'_walls'],data[split+'_goal'],data[split+'_distances']):
            states = flood_oracle(w,int(g.argmax()))
            for k,state in enumerate(states):
                assert np.array_equal(state.reshape(64),(d>=0)&(d<=k))
            assert np.array_equal(states[-1].reshape(64),d>=0)
            oracle_count += 1
    result['independent_flood_oracle_mazes'] = oracle_count
    tensors = {k:torch.from_numpy(data['test_'+k]).to(device) for k in ('walls','goal','distances','start')}
    checkpoints = BASE/'checkpoints'/run.name
    replay,hashes,orders = {},{},{}
    for checkpoint in sorted(checkpoints.glob('*.pt')):
        saved = torch.load(checkpoint,map_location=device,weights_only=False)
        if 'mode' not in saved:
            continue
        model = MazeMLP().to(device)
        model.load_state_dict(saved['model'])
        metrics,outputs = evaluate(model,tensors,saved['mode'],config['eval_steps'])
        expected = np.load(run/(checkpoint.stem+'_states.npz'))['states']
        assert np.array_equal(outputs,expected), checkpoint
        recorded = json.loads((run/(checkpoint.stem+'_metrics.json')).read_text(encoding='utf-8'))
        assert metrics == recorded
        # Independent torch BCE and confusion counts cross-check NumPy metric engine.
        final = torch.from_numpy((data['test_distances']>=0).astype(np.float32))
        max_bce_difference = 0.
        for k in range(config['eval_steps']):
            r = torch.from_numpy(outputs[k,:,128:])
            # Match NumPy's FP64 reduction when confidently wrong repeated A states
            # yield BCE near 30; FP32 summation alone differs by a few e-6.
            bce = F.binary_cross_entropy(r.double(),final.double()).item()
            max_bce_difference = max(max_bce_difference,abs(bce-recorded[k]['final_bce']))
            assert abs(bce-recorded[k]['final_bce']) < 1e-10
            pred,true = r>=.5,final.bool()
            intersection = (pred & true).sum(1).double()
            union = (pred | true).sum(1).clamp_min(1).double()
            assert abs((intersection/union).mean().item()-recorded[k]['macro_iou']) < 1e-12
            assert abs((pred!=true).float().mean().item()-recorded[k]['pixel_error']) < 1e-7
        seed = saved['seed']
        hashes.setdefault(seed,set()).add(saved['initial_sha256'])
        orders.setdefault(seed,set()).add(saved['batch_order_sha256'])
        torch.manual_seed(seed)
        init = MazeMLP().state_dict()
        digest = hashlib.sha256(b''.join(t.numpy().tobytes() for t in init.values())).hexdigest()
        assert digest == saved['initial_sha256']
        expected_steps = config['epochs']*((config['sizes'][0]+config['batch_size']-1)//config['batch_size'])
        assert saved['optimizer_steps']==expected_steps
        replay[checkpoint.stem] = {'max_output_difference':0.,'independent_torch_bce_max_difference':max_bce_difference}
    assert len(replay)==len(config['seeds'])*4
    assert all(len(v)==1 for v in hashes.values()) and all(len(v)==1 for v in orders.values())
    result.update(checkpoints=replay,paired_initial_weights_match=True,paired_batch_orders_match=True)
    after = preservation()
    assert original == after,'Existing H001, EXP001 or shared data modified'
    result.update(preservation_exact=True,preserved_files=len(original))
    return result


if __name__=='__main__':
    import argparse
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    result = verify_run(run,config,torch.device(config['device']),json.loads((run/'preservation_before.json').read_text(encoding='utf-8')))
    (run/'verification_recheck.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verification passed',flush=True)
