"""Strict parsers validated against the actual official archive contents."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from download_benchmark import DATA,MAPS


def parse_map(text):
    lines=text.splitlines()
    if len(lines)<4 or lines[0].strip()!='type octile' or lines[3].strip()!='map':
        raise ValueError('Unsupported map header')
    height=int(lines[1].split()[1]); width=int(lines[2].split()[1])
    if height!=32 or width!=32 or len(lines[4:])!=height or any(len(row)!=width for row in lines[4:]):
        raise ValueError('Expected exact 32x32 map; no resizing permitted')
    terrain=np.array([list(row) for row in lines[4:]])
    symbols=set(terrain.reshape(-1))
    if not symbols<=set('.G@OTS'):
        raise ValueError(f'Unsupported terrain transitions: {symbols-set(".G@OTS")}; water is not silently recoded')
    # G/S are terrain symbols, never the scenario's goal/start.
    walls=np.isin(terrain,list('@OT')).astype(np.uint8)
    return walls,{'width':width,'height':height,'symbols':sorted(symbols),'header_type':'octile'}


def parse_scenario(text,expected_map,archive,member,family):
    lines=text.splitlines()
    if not lines or lines[0].split() not in (['version','1'],['version','1.0']):
        raise ValueError('Unsupported scenario version')
    rows=[]
    for line_number,line in enumerate(lines[1:],2):
        if not line.strip():
            continue
        fields=line.split()
        if len(fields)!=9:
            raise ValueError(f'{member}:{line_number}: expected actual observed 9-field format, got {len(fields)}')
        bucket=int(fields[0]); name=Path(fields[1]).name
        width,height,sx,sy,gx,gy=map(int,fields[2:8])
        cost=float(fields[8])
        if name!=expected_map+'.map' or width!=32 or height!=32 or not np.isfinite(cost) or cost<0:
            raise ValueError(f'Invalid map/cost in {member}:{line_number}')
        rows.append({'map_name':expected_map,'bucket':bucket,'sx':sx,'sy':sy,'gx':gx,'gy':gy,
                     'original_cost':cost,'archive':archive,'member':member,'line':line_number,
                     'scenario_type':family,'scenario_id':member+':'+str(line_number)})
    return rows


def read_official():
    maps,rows,metadata=[],[],[]
    for name in MAPS:
        filename=name+'.map.zip'
        with ZipFile(DATA/'raw'/filename) as z:
            members=[m for m in z.namelist() if not m.endswith('/')]
            if len(members)!=1 or Path(members[0]).name!=name+'.map':
                raise ValueError('Unexpected map archive structure')
            walls,info=parse_map(z.read(members[0]).decode('utf-8'))
        maps.append(walls)
        info.update(name=name,archive=filename,member=members[0])
        metadata.append(info)
        for family in ('even','random'):
            archive=name+f'.map-scen-{family}.zip'
            with ZipFile(DATA/'raw'/archive) as z:
                members=sorted(m for m in z.namelist() if m.endswith('.scen'))
                if len(members)!=25:
                    raise ValueError(f'Expected 25 official files in {archive}, found {len(members)}')
                for member in members:
                    rows.extend(parse_scenario(z.read(member).decode('utf-8'),name,archive,member,family))
    return np.array(maps,dtype=np.uint8),rows,metadata
