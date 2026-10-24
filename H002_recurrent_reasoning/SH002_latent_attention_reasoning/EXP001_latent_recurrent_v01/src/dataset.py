"""Read-only official EXP004 records; new labels are separate run artifacts."""
from zipfile import ZipFile
import hashlib
import numpy as np
from common import DATA,ROOT,read,sha,write
from labels import bfs,dijkstra,directions

MAPS=['maze-32-32-2','maze-32-32-4','random-32-32-10','random-32-32-20','room-32-32-4']


class Dataset:
    def __init__(self,device='cuda'):
        import torch
        self.data=dict(np.load(DATA/'dataset.npz'))
        assert sha(DATA/'dataset.npz')==read(DATA/'complete.json')['dataset_sha256']
        self.ids={k:np.flatnonzero(self.data['split']==i) for i,k in enumerate(('train','val','seen','unseen'))}
        assert [len(self.ids[k]) for k in self.ids]==[25548,3208,3227,20533]
        fields=self.data['fields']; keys=self.data['field_keys']; walls=self.data['walls']
        self.labels=np.stack([directions(walls[m],f) for f,(m,g) in zip(fields,keys)])
        self.correct=torch.from_numpy(self.labels).to(device)
        # Exactly local W/S/G and normalized row/column; no reference metadata in input.
        y,x=np.mgrid[:32,:32]; coords=np.stack((y.reshape(-1)/31,x.reshape(-1)/31),-1).astype(np.float32)
        tokens=np.zeros((len(self.data['start']),1024,5),dtype=np.float32)
        tokens[:,:,0]=walls[self.data['map_id']]; tokens[:,:,3:]=coords
        rows=np.arange(len(tokens)); tokens[rows,self.data['start'],1]=1; tokens[rows,self.data['goal'],2]=1
        self.tokens=torch.from_numpy(tokens).to(device)
        self.field_ids=torch.from_numpy(self.data['field_id'].astype(np.int64)).to(device)
        self.device=device

    def batch(self,ids):
        import torch
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        return self.tokens[ids],self.correct[self.field_ids[ids]]


def audit(run=None):
    from evaluate import decode
    data=dict(np.load(DATA/'dataset.npz')); manifest=read(ROOT/'data/movingai_exp004/download_manifest.json')
    raw=ROOT/'data/movingai_exp004/raw'
    for archive in manifest['files']:
        assert sha(raw/archive['archive'])==archive['sha256']
        with ZipFile(raw/archive['archive']) as z:
            for member in archive['members']: assert hashlib.sha256(z.read(member['name'])).hexdigest()==member['sha256']
    records=read(DATA/'accepted_records.json'); labels=[]
    for m,name in enumerate(MAPS):
        with ZipFile(raw/(name+'.map.zip')) as archive:
            member=next(n for n in archive.namelist() if n.endswith('.map'))
            lines=archive.read(member).decode('utf-8').splitlines()
        assert lines[:4]==['type octile','height 32','width 32','map']
        terrain=np.array([list(row) for row in lines[4:]])
        assert terrain.shape==(32,32) and set(terrain.flat)<=set('.G@OTS')
        assert np.array_equal(np.isin(terrain,list('@OT')).astype(np.uint8).reshape(1024),data['walls'][m])
    for field,(m,g) in zip(data['fields'],data['field_keys']):
        assert np.array_equal(field,bfs(data['walls'][m],int(g)))
        assert np.array_equal(field,dijkstra(data['walls'][m],int(g)))
        lab=directions(data['walls'][m],field)
        assert np.array_equal(lab.any(-1),field>0)
        labels.append(lab)
    labels=np.stack(labels); keys=set(); split_goals={}; oracle=0
    for i,r in enumerate(records):
        m,s,g=int(data['map_id'][i]),int(data['start'][i]),int(data['goal'][i])
        assert tuple(data['field_keys'][data['field_id'][i]])==(m,g)
        assert int(r['field'])==int(data['field_id'][i])
        assert r['map_name']==MAPS[m] and s==r['sy']*32+r['sx'] and g==r['gy']*32+r['gx']
        assert (m,s,g) not in keys; keys.add((m,s,g))
        split_goals.setdefault((m,g),set()).add(int(data['split'][i]))
        # Match source scenario row directly, not only the earlier accepted manifest.
        if i==0 or r['archive']!=last_archive:
            if i: z.close()
            z=ZipFile(raw/r['archive']); last_archive=r['archive']; member_cache={}
        if r['member'] not in member_cache: member_cache[r['member']]=z.read(r['member']).decode('utf-8').splitlines()
        row=member_cache[r['member']][r['line']-1].split()
        assert [int(v) for v in row[4:8]]==[r['sx'],r['sy'],r['gx'],r['gy']]
        assert row[1].split('/')[-1]==MAPS[m]+'.map'
        route=decode(labels[data['field_id'][i]].astype(np.float32),data['walls'][m],s,g)
        assert route['status']=='success' and len(route['path'])-1==data['optimal'][i]>=12
        oracle+=1
    z.close()
    assert all(len(v)==1 for v in split_goals.values())
    assert set(data['map_id'][data['split']==3])=={1,3}
    assert not set(data['map_id'][data['split']<3])&{1,3}
    previous=ROOT/'H002_recurrent_reasoning/SH001_shared_weight_iteration/EXP004_official_pathfinding/results/20261024_001323_main_ff5cf824/used_indices.npz'
    old=dict(np.load(previous))
    for k,name in enumerate(('train','val','test_seen','test_unseen')): assert np.array_equal(np.flatnonzero(data['split']==k),old[name])
    result={'dataset_id':DATA.name,'accepted':len(records),'independent_fields':len(labels),
            'oracle_optimal_rate':oracle/len(records),'coordinates_raw_rows_exact':True,'BFS_Dijkstra_exact':True,
            'EXP004_problem_ids_and_splits_exact':True,'no_goal_split_overlap':True,'raw_archives_exact':True,
            'labels_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),
            'label_origin':'new derived set-valued legal four-way unit-distance decrease labels; not official annotations'}
    if run:
        write(run/'dataset_audit.json',result)
        write(run/'data_hashes.json',{p.name:sha(p) for p in DATA.iterdir() if p.is_file()})
        np.savez_compressed(run/'direction_labels.npz',correct=labels,field_keys=data['field_keys'])
    return result
