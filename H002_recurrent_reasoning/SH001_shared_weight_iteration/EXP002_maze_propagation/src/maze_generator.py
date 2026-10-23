"""BFS labels are confined to data generation/audit; split by wall-map identity."""
from collections import deque
import hashlib
import numpy as np


def neighbors(index):
    r, c = divmod(int(index), 8)
    return [rr * 8 + cc for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
            if 0 <= rr < 8 and 0 <= cc < 8]


def bfs(walls, goal):
    walls = np.asarray(walls).reshape(64)
    distances = np.full(64, -1, dtype=np.int16)
    if walls[goal]:
        raise ValueError('Goal must be open')
    distances[goal] = 0
    queue = deque([int(goal)])
    while queue:
        node = queue.popleft()
        for nxt in neighbors(node):
            if not walls[nxt] and distances[nxt] == -1:
                distances[nxt] = distances[node] + 1
                queue.append(nxt)
    return distances


def canonical_key(walls):
    """Reject duplicate wall maps even under rotations/reflections, regardless of G/S."""
    grid = walls.astype(np.uint8).reshape(8, 8)
    return min(np.packbits(x).tobytes() for k in range(4)
               for x in (np.rot90(grid, k), np.rot90(grid[:, ::-1], k)))


def candidate(rng, family):
    if family == 0:
        walls = (rng.random((8, 8)) < rng.uniform(.20, .43)).astype(np.uint8)
        # A full separating barrier ensures a genuine disconnected open region.
        walls[int(rng.integers(2, 6)), :] = 1
    else:
        # Random DFS tree on a 3x4 coarse grid in rows 0..4; bottom chamber isolated.
        walls = np.ones((8, 8), dtype=np.uint8)
        first = (int(rng.integers(3)), int(rng.integers(4)))
        visited, stack = {first}, [first]
        walls[first[0]*2, first[1]*2] = 0
        while stack:
            r, c = stack[-1]
            choices = [(rr, cc) for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                       if 0 <= rr < 3 and 0 <= cc < 4 and (rr, cc) not in visited]
            if not choices:
                stack.pop()
                continue
            rr, cc = choices[int(rng.integers(len(choices)))]
            walls[r+rr, c+cc] = walls[rr*2, cc*2] = 0
            visited.add((rr, cc))
            stack.append((rr, cc))
        walls[6:8] = (rng.random((2, 8)) < .15).astype(np.uint8)
    walls = np.rot90(walls, int(rng.integers(4)))
    if rng.random() < .5:
        walls = walls[:, ::-1]
    return np.ascontiguousarray(walls.reshape(64))


def generate(config):
    rng = np.random.default_rng(config['data_seed'])
    records, seen, attempts = [], set(), 0
    total = sum(config['sizes'])
    while len(records) < total:
        attempts += 1
        if attempts > total * 200:
            raise RuntimeError('Maze rejection budget exhausted')
        family = len(records) % 2
        walls = candidate(rng, family)
        key = canonical_key(walls)
        if key in seen:
            continue
        opened = np.flatnonzero(walls == 0)
        if not len(opened):
            continue
        # Choose an endpoint of one component, providing nontrivial propagation depth.
        provisional = bfs(walls, int(rng.choice(opened)))
        goal = int(np.argmax(provisional))
        distances = bfs(walls, goal)
        unreachable = (distances < 0) & (walls == 0)
        if (distances.max() < config['long_path_min_eccentricity'] or
                unreachable.sum() < config['minimum_unreachable_open_cells'] or
                unreachable.sum() / len(opened) < config['minimum_unreachable_open_fraction']):
            continue
        records.append((walls, goal, distances, family))
        seen.add(key)
    # Shuffle the whole unique pool BEFORE splitting to avoid generation-order drift.
    order = rng.permutation(total)
    data, offset = {}, 0
    for split, size in zip(('train', 'val', 'test'), config['sizes']):
        subset = [records[i] for i in order[offset:offset+size]]
        w = np.array([r[0] for r in subset], dtype=np.float32)
        g = np.zeros_like(w)
        d = np.array([r[2] for r in subset], dtype=np.int16)
        starts = []
        # Exactly half reachable S, chosen at >=60% of eccentricity and at least 8 away.
        flags = np.arange(size) % 2 == 0
        rng.shuffle(flags)
        for i, (_, goal, dist, _) in enumerate(subset):
            g[i, goal] = 1
            candidates = np.flatnonzero(dist >= max(8, int(np.ceil(.6 * dist.max())))) if flags[i] else np.flatnonzero((dist < 0) & (w[i] == 0))
            starts.append(int(rng.choice(candidates)))
        for key, value in {'walls':w, 'goal':g, 'distances':d,
                           'start':np.array(starts, dtype=np.int64),
                           'family':np.array([r[3] for r in subset], dtype=np.int8)}.items():
            data[split+'_'+key] = value
        offset += size
    return data, {'attempts': attempts, 'accepted': total, 'seed': config['data_seed'],
                  'unique_wall_maps_up_to_dihedral_symmetry': total,
                  'start_policy': 'balanced labels; reachable starts distance >= max(8, ceil(0.6*eccentricity))',
                  'pool_shuffled_before_split': True}


def describe(values):
    values = np.asarray(values)
    return {'min': float(values.min()), 'mean': float(values.mean()), 'max': float(values.max()),
            'percentiles_25_50_75_90': np.percentile(values, [25,50,75,90]).tolist(),
            'histogram': {str(v):int(n) for v,n in zip(*np.unique(values, return_counts=True))}}


def audit(data):
    sets, report = {}, {}
    for split in ('train','val','test'):
        w, g, d, start = [data[split+'_'+k] for k in ('walls','goal','distances','start')]
        sets[split] = {canonical_key(x) for x in w}
        assert len(sets[split]) == len(w)
        assert w.shape == g.shape == d.shape and w.shape[1] == 64
        assert np.all(g.sum(1) == 1) and np.all((w*g).sum(1) == 0)
        assert np.all(w[np.arange(len(w)),start] == 0)
        assert all(np.array_equal(bfs(x,int(y.argmax())),z) for x,y,z in zip(w,g,d))
        reachable = d >= 0
        start_dist = d[np.arange(len(d)), start]
        assert (start_dist >= 0).sum() == (len(d)+1)//2
        eccentricity = d.max(1)
        free_all = w == 0
        all_free_iou = reachable.sum(1) / free_all.sum(1)
        report[split] = {'n':len(w), 'wall_density':describe(w.mean(1)),
                         'reachable_fraction_of_grid':describe(reachable.mean(1)),
                         'reachable_fraction_of_open':describe(all_free_iou),
                         'eccentricity':describe(eccentricity),
                         'reachable_start_shortest_path':describe(start_dist[start_dist >= 0]),
                         'fraction_eccentricity_gt8':float((eccentricity>8).mean()),
                         'fraction_eccentricity_gt16':float((eccentricity>16).mean()),
                         'start_reachable_fraction':float((start_dist>=0).mean()),
                         'all_open_baseline':{'macro_iou':float(all_free_iou.mean()),
                                              'pixel_error':float((free_all != reachable).mean()),
                                              'unreachable_open_false_positive_rate':1.,
                                              'start_accuracy':.5},
                         'zero_baseline_pixel_error':float(reachable.mean()),
                         'family_counts': {str(k):int(v) for k,v in zip(*np.unique(data[split+'_family'],return_counts=True))},
                         'sha256':hashlib.sha256(w.tobytes()+g.tobytes()+d.tobytes()+start.tobytes()).hexdigest()}
        assert np.all((free_all & ~reachable).sum(1) >= 6)
        assert np.all(all_free_iou <= .8 + 1e-7)
        assert (eccentricity > 8).mean() >= .2
        assert (eccentricity > 16).mean() >= .05
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    report['split_wall_map_overlap_even_under_rotations_reflections'] = 0
    return report
