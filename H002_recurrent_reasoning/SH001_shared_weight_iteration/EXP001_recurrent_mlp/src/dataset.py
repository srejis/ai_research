"""Unique clean patterns across splits; fixed independent corruption streams."""
import hashlib
import numpy as np


def pattern_key(x):
    return np.packbits(x.astype(np.uint8)).tobytes()


def generate(config):
    rng = np.random.default_rng(config['pattern_seed'])
    patterns, seen = [], set()
    while len(patterns) < sum(config['sizes']):
        p = np.zeros((8, 8), dtype=np.uint8)
        for _ in range(int(rng.integers(1, 5))):
            kind = int(rng.integers(3))
            r0, r1 = sorted(rng.choice(8, 2, replace=False))
            c0, c1 = sorted(rng.choice(8, 2, replace=False))
            if kind == 0:
                p[r0:r1 + 1, c0] = 1
            elif kind == 1:
                p[r0, c0:c1 + 1] = 1
            else:
                p[r0:r1 + 1, c0] = p[r0:r1 + 1, c1] = 1
                p[r0, c0:c1 + 1] = p[r1, c0:c1 + 1] = 1
        # Every clean target is left-right symmetric: useful but imperfect prior.
        p = np.maximum(p, p[:, ::-1]).reshape(64)
        key = pattern_key(p)
        if key not in seen:
            seen.add(key)
            patterns.append(p)
    data, offset = {}, 0
    for name, size, seed in zip(('train', 'val', 'test'), config['sizes'], config['noise_seeds']):
        y = np.asarray(patterns[offset:offset + size], dtype=np.float32)
        mask = np.random.default_rng(seed).random(y.shape) < config['noise_probability']
        data[name + '_y'] = y
        data[name + '_x'] = np.logical_xor(y, mask).astype(np.float32)
        offset += size
    return data


def audit(data):
    keys = {s: {pattern_key(x) for x in data[s + '_y']} for s in ('train', 'val', 'test')}
    assert not (keys['train'] & keys['val'] or keys['train'] & keys['test'] or keys['val'] & keys['test'])
    result = {'unique_targets': {s: len(k) for s, k in keys.items()}, 'split_overlap': 0}
    for s in keys:
        x, y = data[s + '_x'], data[s + '_y']
        assert len(keys[s]) == len(y)
        assert x.shape == y.shape and x.shape[1] == 64
        # With symmetric targets, a corrupted pair that disagrees remains ambiguous.
        pair_disagreement = x.reshape(-1, 8, 8) != x.reshape(-1, 8, 8)[:, :, ::-1]
        result[s] = {'noise_rate': float((x != y).mean()),
                     'foreground_rate': float(y.mean()),
                     'zero_predictor_error': float(y.mean()),
                     'ambiguous_mirror_pair_fraction': float(pair_disagreement.mean()),
                     'sha256': hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()}
    # Identical noisy inputs with different targets would be directly conflicting labels.
    labels, conflicts = {}, 0
    for s in keys:
        for x, y in zip(data[s + '_x'], data[s + '_y']):
            k, v = pattern_key(x), pattern_key(y)
            conflicts += int(k in labels and labels[k] != v)
            labels[k] = v
    result['observed_conflicting_noisy_inputs'] = conflicts
    return result
