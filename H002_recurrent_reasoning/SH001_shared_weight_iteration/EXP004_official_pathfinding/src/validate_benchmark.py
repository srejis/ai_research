"""Audit every official row, independently verify distances, retain exclusion ledger."""
from collections import Counter
import hashlib
import json
from zipfile import ZipFile
import numpy as np
from download_benchmark import DATA,MAPS,download,digest
from parse_benchmark import read_official
from shortest_path import bfs,dijkstra


def verify_raw():
    manifest=download()
    for archive in manifest['files']:
        path=DATA/'raw'/archive['archive']
        assert digest(path)==archive['sha256']
        with ZipFile(path) as z:
            for member in archive['members']:
                assert hashlib.sha256(z.read(member['name'])).hexdigest()==member['sha256']
    return manifest


def topology(walls):
    grid=walls.reshape(32,32).astype(bool)
    degree=np.zeros((32,32),dtype=int)
    degree[1:]+=~grid[:-1]; degree[:-1]+=~grid[1:]
    degree[:,1:]+=~grid[:,:-1]; degree[:,:-1]+=~grid[:,1:]
    return {'wall_density':float(grid.mean()),'branches':int(((degree>=3)&~grid).sum()),
            'dead_ends':int(((degree==1)&~grid).sum())}


def validate(config,output):
    raw_manifest=verify_raw()
    walls,rows,map_info=read_official()
    fields,field_keys,field_indices,octile=[],[],{},{}
    accepted,ledger,seen=[],[],set()
    counts=Counter(); bymap={name:Counter() for name in MAPS}
    max_original_cost_difference=0.; checked_cost=0
    for raw_index,row in enumerate(rows):
        m=MAPS.index(row['map_name'])
        sx,sy,gx,gy=[row[k] for k in ('sx','sy','gx','gy')]
        reason=None
        if not all(0<=x<32 for x in (sx,sy,gx,gy)):
            reason='coordinate_out_of_bounds'
        elif walls[m,sy,sx] or walls[m,gy,gx]:
            reason='blocked_start_or_goal'
        else:
            start,goal=sy*32+sx,gy*32+gx
            key=(m,goal)
            if key not in field_indices:
                four=bfs(walls[m],goal)
                independent=dijkstra(walls[m],goal)
                if not np.array_equal(four,independent):
                    raise AssertionError(f'BFS/Dijkstra disagreement: {key}')
                field_indices[key]=len(fields)
                fields.append(four)
                field_keys.append(key)
                octile[key]=dijkstra(walls[m],goal,diagonals=True)
            f=field_indices[key]
            d=int(fields[f][start])
            cost_difference=abs(float(octile[key][start])-row['original_cost'])
            if not np.isfinite(cost_difference) or cost_difference>1e-5:
                raise ValueError(f'Original ninth field does not match independently checked no-corner-cut octile cost: {row["scenario_id"]}, difference={cost_difference}')
            checked_cost+=1
            max_original_cost_difference=max(max_original_cost_difference,cost_difference)
            if start==goal:
                reason='identical_start_goal'
            elif d<0:
                reason='unreachable_four_way'
            elif d<config['minimum_four_way_distance']:
                reason='four_way_distance_below_12'
            elif (m,start,goal) in seen:
                reason='duplicate_map_start_goal'
            else:
                seen.add((m,start,goal))
                manhattan=abs(sx-gx)+abs(sy-gy)
                accepted.append({**row,'raw_row_index':raw_index,'map_id':m,'start':start,'goal':goal,
                                 'field':f,'optimal_four_way':d,'manhattan':manhattan,'detour_ratio':d/manhattan})
        label=reason or 'accepted'
        counts[label]+=1; bymap[MAPS[m]][label]+=1
        ledger.append({**row,'raw_row_index':raw_index,'outcome':label})
        if (raw_index+1)%10000==0:
            print(f'AUDIT rows={raw_index+1}/{len(rows)} unique_goal_fields={len(fields)}',flush=True)
    for i,info in enumerate(map_info):
        info.update(topology(walls[i]))
    report={'raw_rows':len(rows),'scenario_files':250,'parse_success_rows':len(rows),'parse_failure_rows':0,
            'actual_format':'version 1; 9 tab/whitespace fields: bucket,map,width,height,sx,sy,gx,gy,original_octile_cost',
            'coordinate_system':'upper-left (0,0); x=column, y=row; flattened index=y*32+x; no rotation or resizing',
            'original_cost_semantics':'verified sqrt(2) diagonals, no corner cutting; NOT used as our four-way optimum',
            'original_octile_cost_rows_checked':checked_cost,'original_cost_max_abs_difference':max_original_cost_difference,
            'our_rule':'four orthogonal neighbors, unit edge cost; BFS independently equals heap Dijkstra on every goal field',
            'unique_goal_fields':len(fields),'counts':dict(counts),'per_map':{k:dict(v) for k,v in bymap.items()},
            'maps':map_info,'raw_archive_sha256':{r['archive']:r['sha256'] for r in raw_manifest['files']}}
    output.mkdir(parents=True,exist_ok=False)
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'row_ledger.jsonl').open('w',encoding='utf-8') as stream:
        for row in ledger:
            stream.write(json.dumps(row)+'\n')
    return walls,np.array(fields,dtype=np.int16),np.array(field_keys,dtype=np.int32),accepted,report
