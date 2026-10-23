"""Run the complete paired-seed experiment; never overwrite a run."""
import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from dataset import generate, audit
from model import RecurrentMLP
from evaluate import evaluate, latency
from verify import preservation, structural, verify_run

BASE = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=BASE / 'configs/experiment.json')
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--smoke', action='store_true', help='Separate small pipeline check; not main experiment')
    args = parser.parse_args()
    if os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('Set PYTHONNOUSERSITE=1 before starting Python')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.device:
        config['device'] = args.device
    if args.smoke:
        config.update(seeds=[42], sizes=[128, 64, 64], epochs=2, latency_warmup=2, latency_repeats=3)
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable: torch={torch.__version__}, build CUDA={torch.version.cuda}. Check NVIDIA driver and CUDA PyTorch build, or explicitly use --device cpu.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + ('_smoke_' if args.smoke else '_main_') + uuid.uuid4().hex[:8]
    run, checkpoints = BASE / 'results' / run_id, BASE / 'checkpoints' / run_id
    run.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)
    print('RUN_ID=' + run_id, flush=True)
    write(run / 'config.json', config)
    shutil.copytree(BASE / 'src', run / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    manifest = preservation()
    write(run / 'preservation_before.json', manifest)
    environment = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                   'numpy': np.__version__, 'cuda_build': torch.version.cuda, 'device': str(device),
                   'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                   'platform': platform.platform(), 'PYTHONNOUSERSITE': os.environ['PYTHONNOUSERSITE'],
                   'precision': 'FP32', 'tf32': False, 'deterministic': True,
                   'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    write(run / 'environment.json', environment)
    write(run / 'structure.json', structural())
    data = generate(config)
    np.savez_compressed(run / 'dataset.npz', **data)
    write(run / 'dataset_audit.json', audit(data))
    tensors = {k: torch.from_numpy(v).to(device) for k, v in data.items()}
    summary = []
    for seed in config['seeds']:
        torch.manual_seed(seed)
        initial = copy.deepcopy(RecurrentMLP().state_dict())
        initial_hash = hashlib.sha256(b''.join(t.numpy().tobytes() for t in initial.values())).hexdigest()
        for mode in ('A', 'B', 'C'):
            name = f'{mode}_seed{seed}'
            model = RecurrentMLP().to(device)
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            generator = torch.Generator().manual_seed(seed + 10000)
            history, optimizer_steps, maximum_grad = [], 0, 0.
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                for epoch in range(config['epochs']):
                    model.train()
                    order = torch.randperm(len(tensors['train_x']), generator=generator).to(device)
                    losses = []
                    epoch_max_grad = 0.
                    for indices in order.split(config['batch_size']):
                        optimizer.zero_grad(set_to_none=True)
                        states = model(tensors['train_x'][indices], 1 if mode == 'A' else config['train_steps'])
                        target = tensors['train_y'][indices]
                        loss = (torch.stack([F.binary_cross_entropy(h, target) for h in states]).mean()
                                if mode == 'C' else F.binary_cross_entropy(states[-1], target))
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite training loss')
                        loss.backward()
                        norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
                        if not np.isfinite(norm):
                            raise FloatingPointError('Nonfinite gradient norm')
                        epoch_max_grad = max(epoch_max_grad, norm)
                        optimizer.step()
                        optimizer_steps += 1
                        losses.append(loss.item())
                    maximum_grad = max(maximum_grad, epoch_max_grad)
                    with torch.no_grad():
                        val = model(tensors['val_x'], 1 if mode == 'A' else config['train_steps'])[-1]
                        val_loss = F.binary_cross_entropy(val, tensors['val_y']).item()
                    history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)),
                                    'val_bce': val_loss, 'gradient_norm_max': epoch_max_grad})
                    if (epoch + 1) % 20 == 0 or epoch == 0:
                        print(f'{name} epoch={epoch + 1} loss={np.mean(losses):.6f} val={val_loss:.6f}', flush=True)
            except Exception as error:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoints / (name + '_FAILED.pt'))
                write(run / (name + '_failure.json'), {'error': repr(error), 'history': history, 'optimizer_steps': optimizer_steps})
                raise
            if device.type == 'cuda':
                torch.cuda.synchronize()
            duration = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'seed': seed, 'mode': mode, 'initial_sha256': initial_hash, 'config': config,
                        'optimizer_steps': optimizer_steps}, checkpoints / (name + '.pt'))
            write(run / (name + '_history.json'), history)
            rows, outputs = evaluate(model, tensors['test_x'], tensors['test_y'], config['eval_steps'])
            np.savez_compressed(run / (name + '_outputs.npz'), states=outputs)
            write(run / (name + '_metrics.json'), rows)
            timings = latency(model, torch.from_numpy(data['test_x'][:config['batch_size']]), device, config)
            write(run / (name + '_latency.json'), timings)
            summary.append({'name': name, 'seed': seed, 'mode': mode, 'initial_sha256': initial_hash,
                            'train_seconds': duration, 'optimizer_steps': optimizer_steps,
                            'training_mapping_calls': optimizer_steps * (1 if mode == 'A' else config['train_steps']),
                            'peak_gpu_allocated_bytes': peak, 'gradient_norm_max': maximum_grad,
                            'metrics': rows, 'latency': timings})
            write(run / 'summary.json', summary)
            print(f'{name} complete {duration:.1f}s; error step1={rows[0]["pixel_error"]:.4f}, step8={rows[7]["pixel_error"]:.4f}', flush=True)
    write(run / 'verification.json', verify_run(run, checkpoints, config, device, manifest))
    from visualize import render
    render(run)
    print('COMPLETE ' + str(run), flush=True)


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
