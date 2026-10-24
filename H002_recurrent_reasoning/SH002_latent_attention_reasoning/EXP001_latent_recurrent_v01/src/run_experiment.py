"""Sequential, durable experiment orchestration. Reuse completed stages explicitly."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from datetime import datetime
from pathlib import Path
import time
import traceback
from common import BASE,DATA,read,write,setup,source_hashes,preservation


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase0',type=Path,required=True); parser.add_argument('--overfit',type=Path,required=True)
    parser.add_argument('--workflow',type=Path); args=parser.parse_args()
    setup(); config=read(BASE/'configs/experiment.json')
    assert [config[k] for k in ('hidden','slots','heads','ffn_hidden','max_steps')]==[128,16,4,512,16]
    workflow=args.workflow or BASE/'results'/('workflow_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    state=read(workflow) if workflow.exists() else {'phase0':str(args.phase0.resolve()),'overfit':str(args.overfit.resolve()),'source_hashes':source_hashes(),'config':config,'started':datetime.now().isoformat()}
    assert state['source_hashes']==source_hashes() and state['config']==config
    write(workflow,state)
    from dataset import audit
    from verify import structure
    from train_solver import train
    from train_halt import train as train_halt
    from diagnose import diagnose
    from visualize import render
    from report import report
    def mark(stage):
        def callback(run): state['active_'+stage]=str(run.resolve()); write(workflow,state)
        return callback
    def active(stage):
        return Path(state['active_'+stage]) if 'active_'+stage in state else None
    def finish(stage,run):
        state[stage]=str(run.resolve()); state.pop('active_'+stage,None); write(workflow,state)
    try:
        assert read(args.phase0/'dataset_audit.json')==audit()
        checks=structure(); write(workflow.with_name(workflow.stem+'_structure.json'),checks)
        preflight=read(args.overfit/'result.json')
        assert all(v['final_loss']<v['initial_loss'] for v in preflight.values())
        state['preflight_strict_pass']=all(v['strict_pass'] for v in preflight.values())
        state['preflight_interpretation']='Strict full overfit passed' if state['preflight_strict_pass'] else 'Only partial trainability established; full-data results remain a limited v0.1 test, not proof that problem complexity caused failure'
        write(workflow,state)
        if 'smoke_solver' not in state:
            prior=active('smoke_solver')
            run=prior if prior and (prior/'complete.json').exists() else train(config,args.phase0,resume=prior,smoke=True,on_started=mark('smoke_solver'))
            finish('smoke_solver',run)
        if 'smoke_halt' not in state:
            prior=active('smoke_halt')
            run=prior if prior and (prior/'complete.json').exists() else train_halt(Path(state['smoke_solver']),config,resume=prior,smoke=True,on_started=mark('smoke_halt'))
            finish('smoke_halt',run)
        if 'solver' not in state:
            prior=active('solver')
            run=prior if prior and (prior/'complete.json').exists() else train(config,args.phase0,resume=prior,on_started=mark('solver'))
            finish('solver',run)
        solver=Path(state['solver'])
        if not (solver/'diagnostics_complete.json').exists(): diagnose(solver)
        if 'halt' not in state:
            prior=active('halt')
            run=prior if prior and (prior/'complete.json').exists() else train_halt(solver,config,resume=prior,on_started=mark('halt'))
            finish('halt',run)
        halt=Path(state['halt']); render(solver,halt); report(solver,halt,state)
        assert preservation()==read(args.phase0/'preservation_before.json')
        state.update(status='complete',finished=datetime.now().isoformat(),preservation_exact=True)
        write(workflow,state); print('WORKFLOW_COMPLETE '+str(workflow),flush=True)
    except BaseException:
        state.update(status='interrupted',traceback=traceback.format_exc()); write(workflow,state); raise


if __name__=='__main__': main()
