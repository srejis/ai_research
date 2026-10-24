"""UTF-8 artifacts, exact provenance, RNG and immutable run management."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid
from datetime import datetime
import numpy as np

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]
DATA=ROOT/'data/movingai_exp004/prepared/3e4d5629b9fe7e71'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def preservation():
    result={}
    for root in (ROOT/'H001_dynamic_network',ROOT/'H002_recurrent_reasoning/SH001_shared_weight_iteration',ROOT/'data'):
        for p in sorted(root.rglob('*')):
            if p.is_file(): result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def setup():
    import torch
    if os.environ.get('PYTHONNOUSERSITE')!='1': raise RuntimeError('PYTHONNOUSERSITE=1 required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda')


def rng_state():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(value):
    import torch
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in value['cuda']])


def model_hash(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items(): h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_hashes(): return {p.name:sha(p) for p in sorted((BASE/'src').glob('*.py'))}


def new_run(phase,config):
    run=BASE/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+phase+'_'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False); (BASE/'checkpoints'/run.name).mkdir(parents=True,exist_ok=False)
    shutil.copytree(BASE/'src',run/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(run/'source_hashes.json',source_hashes()); write(run/'config.json',config)
    write(run/'environment.json',environment())
    print('RUN_ID='+run.name,flush=True)
    return run


def environment():
    import torch,platform
    return {'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform(),
            'FP32':True,'TF32':False,'deterministic':True,'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def checkpoint_path(run,name): return BASE/'checkpoints'/run.name/(name+'.pt')


def save_checkpoint(path,value):
    import torch
    tmp=path.with_suffix('.tmp'); torch.save(value,tmp); tmp.replace(path)


def checked_resume(run,config):
    assert not (run/'complete.json').exists(), 'Completed runs cannot resume'
    assert read(run/'config.json')==config
    assert read(run/'source_hashes.json')==source_hashes(), 'Source version changed; create a separate run'
    assert read(run/'data_hashes.json')=={p.name:sha(p) for p in DATA.iterdir() if p.is_file()}
    events=read(run/'resume_events.json') if (run/'resume_events.json').exists() else []
    events.append({'time':datetime.now().isoformat(),'source_exact':True,'config_exact':True,'data_exact':True})
    write(run/'resume_events.json',events)
