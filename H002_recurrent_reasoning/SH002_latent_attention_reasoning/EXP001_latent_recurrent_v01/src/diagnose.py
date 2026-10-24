"""Evaluation-only intermediate readouts, state/attention statistics and history perturbations."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import time
import numpy as np
import torch
from common import read,write,setup,checkpoint_path,model_hash
from dataset import Dataset
from model import Solver
from evaluate import evaluate


def diagnose(run):
    config=read(run/'config.json'); ds=Dataset('cuda'); ids=np.load(run/'used_problem_ids.npz')['evaluation']
    output=[]
    for seed in config['seeds']:
        model=Solver().to(ds.device); saved=torch.load(checkpoint_path(run,f'B_seed{seed}_best'),map_location=ds.device,weights_only=False)
        model.load_state_dict(saved['model']); model.eval(); before=model_hash(model)
        for steps in (1,2,4,8):
            path=run/f'diagnostic_B_seed{seed}_T{steps}_groups.json'
            if not path.exists():
                rows,_,_=evaluate(model,ds,ids,steps,True,config['eval_batch'],save=run/f'diagnostic_B_seed{seed}_T{steps}')
            else: rows=read(path)
            output.append({'seed':seed,'kind':'truncated_fixed','steps':steps,'groups':rows})
        for variant in ('remove','zero','reverse','shuffle','other'):
            path=run/f'diagnostic_B_seed{seed}_{variant}_groups.json'
            if not path.exists(): rows,_,_=evaluate(model,ds,ids,16,True,config['eval_batch'],variant=variant,save=run/f'diagnostic_B_seed{seed}_{variant}')
            else: rows=read(path)
            output.append({'seed':seed,'kind':'history_perturbation','variant':variant,'groups':rows})
        selected=[]; rng=np.random.default_rng(2026)
        for split in (2,3):
            for band in range(4):
                available=ids[(ds.data['split'][ids]==split)&(ds.data['band'][ids]==band)]
                selected.extend(rng.choice(available,min(32,len(available)),replace=False).tolist())
        selected=np.array(selected,dtype=int)
        if seed==config['seeds'][0]:
            # Include same-problem A/B and B/C disagreements for honest representative figures.
            metrics={m:dict(np.load(run/f'{m}_seed{seed}_metrics.npz')) for m in ('A','B','C')}
            choices={}
            for name,mask in [('B_only_vs_A',(metrics['B']['success']>0)&(metrics['A']['success']==0)),
                              ('A_only_vs_B',(metrics['A']['success']>0)&(metrics['B']['success']==0)),
                              ('B_only_vs_C',(metrics['B']['success']>0)&(metrics['C']['success']==0)),
                              ('C_only_vs_B',(metrics['C']['success']>0)&(metrics['B']['success']==0)),
                              ('B_failure',metrics['B']['success']==0)]:
                match=np.flatnonzero(mask)
                if len(match): choices[name]=int(ids[match[0]])
            choices.update(shortest=int(ids[np.argmin(ds.data['optimal'][ids])]),longest=int(ids[np.argmax(ds.data['optimal'][ids])]))
            write(run/'example_selection.json',choices)
            selected=np.unique(np.concatenate((selected,np.array(list(choices.values()),dtype=int))))
        states=[]; change=[]; source=[]; history=[]; intermediate=[]
        with torch.no_grad():
            for start in range(0,len(selected),config['eval_batch']):
                x,_=ds.batch(selected[start:start+config['eval_batch']]); _,hs,trace,memory=model(x,16,True,trace=True)
                states.append(torch.stack(hs,1).cpu().numpy())
                change.append(torch.stack([v['change'] for v in trace],1).cpu().numpy())
                source.append(torch.stack([v['input_attention'] for v in trace],1).cpu().numpy())
                history.append(torch.stack([v['history_attention'] for v in trace],1).cpu().numpy())
        h=np.concatenate(states); change=np.concatenate(change); source=np.concatenate(source); history=np.concatenate(history)
        assert np.allclose(source+history,1.,atol=1e-5) and np.all(history[:,0]==0.)
        # Compare successive and two-apart changes as convergence/oscillation diagnostics only.
        np.savez_compressed(run/f'B_seed{seed}_latent_diagnostics.npz',ids=selected,states=h,change=change,
                            input_attention=source,history_attention=history,
                            two_step_change=np.sqrt(np.mean((h[:,2:]-h[:,:-2])**2,axis=(2,3))))
        assert model_hash(model)==before
        write(run/'diagnostic_results.json',output)
    write(run/'diagnostics_complete.json',{'complete':True,'training_loss_unchanged':True,
          'history_variants':'remove=no history; zero=zero values plus learned tags; reverse/reindex time; shuffle=cyclic one-slot reindex; other=cyclic batch scenario replacement',
          'attention_caveat':'weights are descriptive input allocation, not semantic explanation of thoughts'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=__import__('pathlib').Path); args=parser.parse_args()
    setup(); diagnose(args.run)
