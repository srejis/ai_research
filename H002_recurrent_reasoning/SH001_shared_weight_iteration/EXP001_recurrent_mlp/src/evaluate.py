"""All-step test metrics and synchronized wall-clock latency measurements."""
import time
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate(model, x, y, steps=32):
    model.eval()
    states = model(x, steps)
    rows, previous, previous_error = [], x, (x.ge(.5) != y.bool()).float().mean(1)
    for k, h in enumerate(states, 1):
        error = (h.ge(.5) != y.bool()).float().mean(1)
        change = (h - previous).abs()
        cycle = (h - states[k - 3]).abs().mean(1) if k >= 3 else torch.full_like(error, float('nan'))
        rows.append({'step': k, 'bce': F.binary_cross_entropy(h, y).item(),
                     'pixel_error': error.mean().item(), 'pixel_accuracy': 1 - error.mean().item(),
                     'exact_recovery': (error == 0).float().mean().item(),
                     'improved': (error < previous_error).float().mean().item(),
                     'worsened': (error > previous_error).float().mean().item(),
                     'unchanged': (error == previous_error).float().mean().item(),
                     'change_mean': change.mean().item(), 'change_max': change.max().item(),
                     'nonfinite': int((~torch.isfinite(h)).sum().item()),
                     'period2_fraction': ((cycle < 1e-5) & (change.mean(1) > 1e-4)).float().mean().item(),
                     'outside_unit_interval': int(((h < 0) | (h > 1)).sum().item())})
        previous, previous_error = h, error
    return rows, torch.stack(states).cpu().numpy()


@torch.no_grad()
def latency(model, host_x, device, config):
    """Model-only excludes H2D; end-to-end includes H2D + D2H output transfer."""
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    result = []
    x = host_x.to(device)
    for steps in config['selected_steps']:
        for _ in range(config['latency_warmup']):
            model(x, steps)
        sync()
        measurements = {}
        for mode in ('model_only', 'end_to_end'):
            times = []
            for _ in range(config['latency_repeats']):
                sync()
                start = time.perf_counter()
                if mode == 'end_to_end':
                    output = model(host_x.to(device), steps)[-1].cpu()
                else:
                    output = model(x, steps)[-1]
                sync()
                times.append((time.perf_counter() - start) * 1000)
            measurements[mode + '_ms_mean'] = float(np.mean(times))
            measurements[mode + '_ms_std'] = float(np.std(times))
        result.append({'step': steps, 'batch_size': len(host_x), **measurements})
    return result
