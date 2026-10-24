"""Independent dataset, architecture, BPTT, decoder, halting and preservation checks."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from common import BASE,DATA,read,write,preservation,source_hashes,sha,model_hash,seed_all


def structure(device='cuda'):
    import torch
    from model import Solver,Halt,parameters
    from labels import policy_loss
    from evaluate import decode
    seed_all(42); solver=Solver().to(device)
    x=torch.zeros(2,1024,5,device=device); x[:,33,1]=1; x[:,55,2]=1
    y,c=torch.meshgrid(torch.arange(32,device=device),torch.arange(32,device=device),indexing='ij')
    x[:,:,3]=y.flatten()/31; x[:,:,4]=c.flatten()/31
    calls=[]
    hook=solver.core.register_forward_hook(lambda module,args,out:calls.append((id(module),args[1][0].shape[2])))
    head_calls=[]; hh=solver.readout.register_forward_hook(lambda *args:head_calls.append(1))
    logits,states,_,memory=solver(x,16,True,return_states=True)
    assert logits.shape==(2,1024,4) and memory.shape==(2,1024,128)
    assert len(calls)==16 and len(set(c[0] for c in calls))==1 and [c[1] for c in calls]==[1024+16*t for t in range(16)]
    assert len(head_calls)==1 and all(h.shape==(2,16,128) for h in states)
    for h in states: h.retain_grad()
    correct=torch.zeros_like(logits,dtype=torch.bool); correct[:,:,1]=True; correct[:,::2,2]=True
    objective=policy_loss(logits,correct).mean(); objective.backward()
    gradient=[float(h.grad.abs().sum()) for h in states]
    assert all(np.isfinite(g) and g>0 for g in gradient)
    for module in (solver.encoder,solver.core.cross,solver.core.self_attention,solver.core.mlp,solver.readout):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(float(p.grad.abs().sum()) for p in module.parameters())>0
    assert solver.initial.grad.abs().sum()>0
    assert solver.row.weight.grad.abs().sum()>0 and solver.column.weight.grad.abs().sum()>0
    assert solver.time.grad[:15].abs().sum()>0 and solver.slot.grad.abs().sum()>0
    # Explicitly isolate history gradient: Q-state detached but an old state remains live in K/V.
    solver.zero_grad(set_to_none=True)
    h=torch.randn(2,16,128,device=device,requires_grad=True)
    kv=solver.history_kv(h,0); probe,_=solver.core(h.detach(),kv); probe.square().mean().backward()
    assert h.grad is not None and h.grad.abs().sum()>0
    history_gradient=float(h.grad.abs().sum())
    hook.remove(); hh.remove()
    # Encoder memory is never updated by the core; call receives projected read-only X slices.
    with torch.no_grad():
        a=solver.encode(x).clone(); solver(x,16,True); assert torch.equal(a,solver.encode(x))
    # Exact multiple-optimal-direction loss: uniform logits, 2 allowed -> log(2).
    zero=torch.zeros(1,1,4,device=device); good=torch.tensor([[[True,True,False,False]]],device=device)
    assert abs(float(policy_loss(zero,good))-np.log(2))<1e-6
    wall=np.zeros(1024,dtype=np.uint8); policy=np.zeros((1024,4)); policy[:,0]=1
    assert decode(policy,wall,0,1023)['status']=='out_of_bounds'
    wall[1]=1; policy[:]=0; policy[:,3]=1
    assert decode(policy,wall,0,1023)['status']=='wall'
    policy[:]=0; policy[:,3]=1; policy[1]=[0,0,2,0]; wall[1]=0
    assert decode(policy,wall,0,1023)['status']=='revisit'
    policy[0]=np.nan; assert decode(policy,wall,0,1023)['status']=='nonfinite'
    # No graph search in model/policy decoder, and no labels enter model arguments.
    import ast,inspect
    tree=ast.parse(inspect.getsource(decode))
    assert not any(isinstance(n,ast.Name) and n.id.lower() in {'bfs','dijkstra','astar','target','correct'} for n in ast.walk(tree))
    # Adaptive batch must stop computing core immediately after unconditional stop.
    halt=Halt().to(device)
    for p in halt.parameters(): p.data.zero_()
    halt.net[-1].bias.data.fill_(30)
    with torch.no_grad():
        _,info=solver.adaptive(x,halt,trace=True)
    assert info['steps'].eq(1).all() and info['active_core_batch_sizes']==[2] and not info['forced'].any()
    halt.net[-1].bias.data.fill_(-30)
    with torch.no_grad(): _,info=solver.adaptive(x,halt,trace=True)
    assert info['steps'].eq(16).all() and info['forced'].all() and len(info['active_core_batch_sizes'])==16
    class MixedHalt(torch.nn.Module):
        def forward(self,h):
            value=torch.full((len(h),),-30.,device=h.device)
            if len(h)==2: value[0]=30.
            return value
    mixed_calls=[]
    mixed_hook=solver.core.register_forward_hook(lambda module,args,out:mixed_calls.append(len(args[0])))
    with torch.no_grad(): _,mixed=solver.adaptive(x,MixedHalt().to(device))
    mixed_hook.remove()
    assert mixed['steps'].tolist()==[1,16] and mixed['forced'].tolist()==[False,True]
    assert mixed_calls==[2]+[1]*15, 'Stopped example must not execute later core iterations'
    # Log-probability includes the actual Bernoulli actions; baseline is independent of sampled action.
    halt.net[-1].bias.data.zero_()
    for p in solver.parameters(): p.requires_grad_(False); p.grad=None
    frozen=model_hash(solver)
    _,info=solver.adaptive(x,halt,stochastic=True,trace=True)
    manual=np.zeros(2)
    for event in info['events']:
        for i,p,stop,lp in zip(event['ids'],event['p'],event['stop'],event['log_probability']):
            assert abs(lp-np.log(p if stop else 1-p))<1e-5; manual[i]+=lp
    assert np.allclose(manual,info['log_probability'].detach().cpu().numpy(),atol=1e-5)
    cost=torch.tensor([.7,1.2],device=device); baseline=torch.tensor(.4,device=device)
    pg=((cost-baseline).detach()*info['log_probability']).mean(); pg.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in halt.parameters())
    assert model_hash(solver)==frozen and all(p.grad is None for p in solver.parameters())
    return {'parameters':parameters(solver),'states_shape':[16,128],'input_shape':[1024,5],'memory_shape':[1024,128],
            'shared_core_calls':16,'memory_lengths':[1024+16*t for t in range(16)],'final_head_calls':1,
            'all_state_gradient_l1':gradient,'isolated_history_gradient_l1':history_gradient,
            'all_attention_gradients':True,'X_read_only':True,'final_loss_only':True,'no_search_decoder':True,
            'unmasked_illegal_fails':True,'multiple_optimal_loss_exact':True,'halt_actions_logprobs_exact':True,
            'halt_1_and_forced_16_tested':True,'no_core_after_stop':True,'frozen_solver_exact':True,
            'mixed_batch_stopped_examples_excluded':True,'mixed_batch_core_batch_sizes':mixed_calls}


def phase0():
    from dataset import audit
    run=BASE/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_phase0')
    run.mkdir(parents=True,exist_ok=False)
    print('PHASE0='+str(run),flush=True)
    before=preservation(); write(run/'preservation_before.json',before)
    print('PRESERVATION_FILES='+str(len(before)),flush=True)
    result=audit(run); write(run/'phase0_data_complete.json',result)
    print('DATA_ORACLE_COMPLETE',flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser(); parser.add_argument('--data-only',action='store_true'); parser.add_argument('--structure',action='store_true')
    args=parser.parse_args()
    if args.data_only: phase0()
    elif args.structure:
        from common import setup
        setup(); result=structure(); write(BASE/'results/structure_check.json',result); print(json.dumps(result),flush=True)
