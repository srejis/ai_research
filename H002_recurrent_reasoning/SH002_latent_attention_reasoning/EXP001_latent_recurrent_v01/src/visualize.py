"""Native scientific plots from actual official routes, latent diagnostics and stopping."""
import argparse
from pathlib import Path
import hashlib
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import BASE,DATA,read,write
from labels import directions,MOVES
from evaluate import decode
from dataset import MAPS


def render(solver_run,halt_run,destination=None):
    summary=read(solver_run/'summary.json'); halts=read(halt_run/'summary.json'); config=read(solver_run/'config.json')
    data=dict(np.load(DATA/'dataset.npz')); records=read(DATA/'accepted_records.json')
    destination=Path(destination) if destination is not None else solver_run/'figures'
    destination.mkdir(exist_ok=True)
    def export(fig,name):
        fig.savefig(destination/name,dpi=150,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for axis,split in zip(axes,('seen','unseen')):
        for mode in ('A','B','C'):
            selected=[r for r in summary if r['mode']==mode]
            values=np.array([[r['groups'][f'{split}/Q{k}']['success'] for k in range(1,5)] for r in selected])
            axis.errorbar(range(1,5),values.mean(0),yerr=values.std(0,ddof=1 if len(values)>1 else 0),label=mode,marker='o',capsize=3)
        axis.set(title=split,xlabel='Training-defined four-way distance quartile',ylabel='Actual path success'); axis.set_xticks(range(1,5)); axis.legend(); axis.grid(alpha=.2)
    fig.suptitle('Final-only latent attention policy | no illegal-action masking or search repair'); export(fig,'success_by_difficulty.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for r in summary:
        h=read(solver_run/f'{r["mode"]}_seed{r["seed"]}_history.json')
        axes[0].plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label=f'{r["mode"]} seed{r["seed"]}')
        valid=[x for x in h if 'validation' in x]
        axes[1].plot([x['epoch'] for x in valid],[x['validation']['success'] for x in valid],label=f'{r["mode"]} seed{r["seed"]}')
    axes[0].set(title='Final policy training loss',xlabel='Epoch',ylabel='Set-valued policy NLL')
    axes[1].set(title='Validation actual completion',xlabel='Epoch',ylabel='Success'); axes[1].legend(fontsize=7,ncol=3)
    export(fig,'training_validation.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for seed in config['seeds']:
        values=dict(np.load(solver_run/f'B_seed{seed}_latent_diagnostics.npz'))
        axes[0].plot(range(1,17),values['change'].mean(0),label=f'seed{seed}')
        axes[1].plot(range(1,17),values['history_attention'].mean(0),label=f'history seed{seed}')
        axes[1].plot(range(1,17),values['input_attention'].mean(0),'--',label=f'input seed{seed}')
    axes[0].set(title='B latent RMS change (diagnostic only)',xlabel='Iteration',ylabel='RMS state delta')
    axes[0].legend(fontsize=8)
    axes[1].set(title='B attention allocation (not thought meaning)',xlabel='Iteration',ylabel='Attention mass'); axes[1].legend(fontsize=7)
    export(fig,'latent_attention.png')
    diagnostics=read(solver_run/'diagnostic_results.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for axis,split in zip(axes,('seen','unseen')):
        for seed in config['seeds']:
            rows=[r for r in diagnostics if r['seed']==seed and r['kind']=='truncated_fixed']
            fixed=next(r for r in summary if r['seed']==seed and r['mode']=='B')
            axis.plot([r['steps'] for r in rows]+[16],[r['groups'][split]['success'] for r in rows]+[fixed['groups'][split]['success']],marker='o',label=f'seed{seed}')
        axis.set(title=split+' | same B checkpoint truncated',xlabel='Forced iterations',ylabel='Actual success'); axis.legend()
    export(fig,'fixed_iteration_diagnostic.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for s in halts:
        for r in s['candidates']:
            group=r['groups']['seen']; axes[0].plot(range(1,17),group['step_histogram'],label=f's{r["seed"]} c{r["penalty"]}')
            axes[1].scatter(group['steps'],group['success'],marker='o' if r['adoptable'] else 'x',label=f's{r["seed"]} c{r["penalty"]}')
    for r in summary:
        if r['mode']=='B': axes[1].scatter(16,r['groups']['seen']['success'],marker='s',c='black')
    axes[0].set(title='Actual adaptive stops | seen',xlabel='Iteration',ylabel='Number of problems')
    axes[1].set(title='Quality vs adaptive computation | seen',xlabel='Mean core iterations',ylabel='Path success'); axes[1].legend(fontsize=7,ncol=2)
    export(fig,'stopping_quality.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for s in halts:
        for r in s['candidates']:
            axes[0].plot(range(1,5),[r['groups'][f'seen/Q{k}']['steps'] for k in range(1,5)],marker='o',label=f's{r["seed"]} c{r["penalty"]}')
            axes[1].plot(range(1,5),[r['groups'][f'seen/Q{k}']['success'] for k in range(1,5)],marker='o')
    axes[0].set(title='Adaptive compute by difficulty',xlabel='Quartile',ylabel='Mean iterations'); axes[0].legend(fontsize=6,ncol=2)
    axes[1].set(title='Adaptive success by difficulty',xlabel='Quartile',ylabel='Actual success'); export(fig,'stopping_difficulty.png')
    choices=read(solver_run/'example_selection.json'); seed=config['seeds'][0]
    paths={mode:read(solver_run/f'{mode}_seed{seed}_paths.json') for mode in ('A','B','C')}
    by_id={m:{r['record_index']:r for r in routes} for m,routes in paths.items()}
    perturb={v:{r['record_index']:r for r in read(solver_run/f'diagnostic_B_seed{seed}_{v}_paths.json')} for v in ('remove','zero','reverse')}
    payload=[]
    def xy(path): return np.array(path)%32,np.array(path)//32
    for i in sorted(set(choices.values())):
        record=records[i]; walls=data['walls'][data['map_id'][i]]; s=int(data['start'][i]); g=int(data['goal'][i]); field=data['fields'][data['field_id'][i]]
        oracle=decode(directions(walls,field).astype(float),walls,s,g); assert oracle['status']=='success'
        fig,axes=plt.subplots(2,4,figsize=(17,9)); cases=[('Oracle',oracle)]+[(m,by_id[m][i]) for m in ('A','B','C')]+[('B '+v,perturb[v][i]) for v in perturb]
        axes.flat[-1].axis('off')
        for axis,(label,r) in zip(axes.flat,cases):
            axis.imshow(walls.reshape(32,32),cmap='gray_r',origin='upper',vmin=0,vmax=1,interpolation='nearest')
            a=axis.scatter(s%32,s//32,c='deepskyblue',edgecolors='black',s=25,zorder=5)
            b=axis.scatter(g%32,g//32,c='gold',edgecolors='black',marker='*',s=60,zorder=5)
            assert np.array_equal(a.get_offsets()[0],[s%32,s//32]) and np.array_equal(b.get_offsets()[0],[g%32,g//32])
            x,y=xy(r['path']); line,=axis.plot(x,y,color='seagreen' if r['status']=='success' else 'crimson',linewidth=1.7)
            assert np.array_equal(line.get_xdata(),x) and np.array_equal(line.get_ydata(),y)
            axis.set_title(f'{label}: {r["status"]}\nL={len(r["path"])-1}, optimum={data["optimal"][i]}, T={r.get("steps",0)}',fontsize=9)
        fig.suptitle(f'{record["map_name"]} | {record["scenario_id"]}\nS=({s%32},{s//32}) G=({g%32},{g//32}) | seed={seed} | optimal={data["optimal"][i]} | no search repair',fontsize=11)
        export(fig,f'case_{i}_paths.png')
        payload.append({'record_index':i,'map':record['map_name'],'scenario_id':record['scenario_id'],'S':[s%32,s//32],'G':[g%32,g//32],
                        'wall_sha256':hashlib.sha256(walls.tobytes()).hexdigest(),'oracle_length':len(oracle['path'])-1,'image':f'case_{i}_paths.png'})
    write(destination/'provenance.json',{'solver_run':solver_run.name,'halt_run':halt_run.name,'samples':payload,
          'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.png')},
          'official_coordinates_and_line_coordinates_verified':True})
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('solver_run',type=Path); parser.add_argument('halt_run',type=Path)
    args=parser.parse_args(); render(args.solver_run,args.halt_run)
