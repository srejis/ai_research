"""Structural/BPTT checks, independent checkpoint replay and preservation audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import RecurrentMLP
from dataset import audit, generate
from evaluate import evaluate


ROOT = Path(__file__).resolve().parents[4]


def preservation():
    files = list((ROOT / 'H001_dynamic_network').rglob('*')) + list((ROOT / 'data').rglob('*'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.is_file()}


def structural():
    torch.manual_seed(0)
    model = RecurrentMLP()
    assert sum(p.numel() for p in model.parameters()) == 16576
    x, y = torch.rand(3, 64), torch.rand(3, 64)
    ids = tuple(id(p) for p in model.parameters())
    calls = []
    hook = model.net.register_forward_hook(lambda module, args, out: calls.append(tuple(id(p) for p in module.parameters())))
    states = model(x, 8)
    hook.remove()
    assert len(calls) == 8 and all(c == ids for c in calls)
    assert all(h.shape == (3, 64) for h in states)
    for h in states:
        h.retain_grad()
    F.binary_cross_entropy(states[-1], y).backward()
    gradients = [h.grad.abs().sum().item() for h in states]
    assert all(np.isfinite(g) and g > 0 for g in gradients)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model(x, 32)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    return {'parameters': 16576, 'shared_calls': len(calls), 'state_gradient_l1': gradients,
            'inference_weights_unchanged': True}


def verify_run(run, checkpoints, config, device, original_manifest):
    data = dict(np.load(run / 'dataset.npz'))
    regenerated = generate(config)
    assert all(np.array_equal(data[k], regenerated[k]) for k in data)
    result = {'structure': structural(), 'dataset': audit(data), 'checkpoints': {}}
    initial_hashes = {}
    x, y = [torch.from_numpy(data['test_' + name]).to(device) for name in ('x', 'y')]
    for checkpoint in sorted(checkpoints.glob('*.pt')):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model = RecurrentMLP().to(device)
        model.load_state_dict(saved['model'])
        initial_hashes.setdefault(saved['seed'], set()).add(saved['initial_sha256'])
        _, outputs = evaluate(model, x, y, config['eval_steps'])
        expected = np.load(run / (checkpoint.stem + '_outputs.npz'))['states']
        difference = float(np.max(np.abs(outputs - expected)))
        assert np.allclose(outputs, expected, rtol=1e-5, atol=1e-6), checkpoint
        metrics = json.loads((run / (checkpoint.stem + '_metrics.json')).read_text(encoding='utf-8'))
        target = data['test_y'].astype(np.float64)
        independent_error = ((outputs >= .5) != target[None].astype(bool)).mean(axis=(1, 2))
        probabilities = outputs.astype(np.float64)
        with np.errstate(divide='ignore'):
            log_p = np.maximum(np.log(probabilities), -100)
            log_not_p = np.maximum(np.log1p(-probabilities), -100)
        independent_bce = -(target[None] * log_p + (1 - target[None]) * log_not_p).mean(axis=(1, 2))
        assert np.allclose(independent_error, [r['pixel_error'] for r in metrics], atol=1e-7)
        assert np.allclose(independent_bce, [r['bce'] for r in metrics], atol=1e-6)
        expected_steps = config['epochs'] * ((config['sizes'][0] + config['batch_size'] - 1) // config['batch_size'])
        assert saved['optimizer_steps'] == expected_steps
        result['checkpoints'][checkpoint.stem] = {'max_output_difference': difference,
                                                'independent_numpy_metrics': True}
    assert len(result['checkpoints']) == 3 * len(config['seeds'])
    assert all(len(hashes) == 1 for hashes in initial_hashes.values())
    result['paired_initial_weights_match'] = True
    result['preserved_files'] = len(original_manifest)
    assert preservation() == original_manifest, 'H001 or shared data changed'
    result['preservation_exact'] = True
    return result


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    original = json.loads((run / 'preservation_before.json').read_text(encoding='utf-8'))
    result = verify_run(run, run.parents[1] / 'checkpoints' / run.name, config,
                        torch.device(config['device']), original)
    (run / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Verification passed', flush=True)
