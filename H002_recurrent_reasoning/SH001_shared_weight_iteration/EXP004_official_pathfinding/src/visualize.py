"""Actual line paths on unchanged official coordinates; no image-only surrogate score."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_dataset import load_dataset
from download_benchmark import MAPS
from evaluate import decode,legal_path

MODES=['A','A_large','A_flops','B','C','D','E']


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def render(run):
    run=Path(run); rows=read(run/'aggregate.json'); config=read(run/'config.json')
    summary=read(run/'summary.json'); ablations=read(run/'ablation_aggregate.json')
    for metric,label in [('success','Path success fraction'),('distance_mae','Distance field MAE (unit steps)'),('spl','Success-weighted optimal/actual cost')]:
        fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True)
        for axis,split in zip(axes,('seen','unseen')):
            for mode in MODES:
                values=[next(r for r in rows if r['mode']==mode and r['group']==f'{split}/Q{k}') for k in range(1,5)]
                mean=[r.get(metric+'_mean',np.nan) for r in values]; std=[r.get(metric+'_std',np.nan) for r in values]
                axis.errorbar(range(4),mean,yerr=std,label=mode,marker='o',capsize=2)
            axis.set_xticks(range(4),['Q1','Q2','Q3','Q4'])
            axis.set(title=split+' maps',xlabel='Four-way distance quartiles fixed from training',ylabel=label)
            axis.grid(alpha=.2)
        axes[-1].legend(ncol=2,fontsize=8)
        fig.suptitle('Official MAPF-derived single-agent task; strict greedy descent; no search repair')
        fig.tight_layout(); fig.savefig(run/(metric+'_by_difficulty.png'),dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for axis,split in zip(axes,('seen','unseen')):
        variants=[r for r in ablations if r['group']==split]
        baseline=next(r['success_mean'] for r in rows if r['mode']=='D' and r['group']==split)
        axis.bar(range(len(variants)),[r['success_mean']-baseline for r in variants])
        axis.set_xticks(range(len(variants)),[r['variant'] for r in variants],rotation=65,ha='right',fontsize=8)
        axis.set(title=split,ylabel='Success difference vs normal trained D')
        axis.axhline(0,color='black',linewidth=.8)
    fig.tight_layout(); fig.savefig(run/'history_interventions.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for mode in MODES:
        records=[r for r in summary if r['mode']==mode]
        success=next(r['success_mean'] for r in rows if r['mode']==mode and r['group']=='seen')
        costs=[np.mean([r['system_train_seconds'] for r in records]),np.mean([r['latency']['model_and_decoder_ms_mean'] for r in records])]
        for axis,cost in zip(axes,costs): axis.scatter(cost,success); axis.annotate(mode,(cost,success))
    axes[0].set(xlabel='System training seconds incl core; cache separate',ylabel='Seen-map path success')
    axes[1].set(xlabel='Score + transfers + greedy decoder ms / batch',ylabel='Seen-map path success')
    fig.tight_layout(); fig.savefig(run/'quality_cost.png',dpi=160); plt.close(fig)
    with (run/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        columns=sorted({k for row in rows for k in row})
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    panels(run,config)
    report(run,rows,summary,ablations,config)


def panels(run,config):
    ref=read(run/'dataset_reference.json'); data=load_dataset(ref['path'])
    accepted=read(Path(ref['path'])/'accepted_records.json')
    ids=np.load(run/'used_indices.npz')['evaluation']; seed=config['seeds'][0]
    paths={m:read(run/f'{m}_seed{seed}_paths.json') for m in MODES}
    scores={m:np.load(run/f'{m}_seed{seed}_scores.npy',mmap_mode='r') for m in MODES}
    h=np.load(run/f'core_seed{seed}_evaluation_history.npy',mmap_mode='r')
    selected={'shortest':int(data['optimal'][ids].argmin()),'longest':int(data['optimal'][ids].argmax())}
    for m in range(5):
        found=np.flatnonzero(data['map_id'][ids]==m)
        if len(found): selected['map_'+str(m)]=int(found[0])
    d=np.array([r['status']=='success' for r in paths['D']]); c=np.array([r['status']=='success' for r in paths['C']])
    for name,mask in [('D_only_success',d&~c),('C_only_success',c&~d),('D_failure',~d)]:
        found=np.flatnonzero(mask)
        if len(found): selected[name]=int(found[0])
    (run/'example_selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    images=run/'examples'; images.mkdir(exist_ok=True)
    payload=[]
    def base(axis,walls,s,g):
        axis.imshow(walls.reshape(32,32),cmap='gray_r',vmin=0,vmax=1,origin='upper',interpolation='nearest')
        sy,sx=divmod(s,32); gy,gx=divmod(g,32)
        a=axis.scatter([sx],[sy],c='#22a7f0',s=35,marker='o',edgecolors='black',linewidths=.4,zorder=4)
        b=axis.scatter([gx],[gy],c='#ffd227',s=55,marker='*',edgecolors='black',linewidths=.4,zorder=5)
        assert np.array_equal(a.get_offsets()[0],[sx,sy]) and np.array_equal(b.get_offsets()[0],[gx,gy])
        axis.set_xlim(-.5,31.5); axis.set_ylim(31.5,-.5)
        axis.set_xticks([0,8,16,24,31]); axis.set_yticks([0,8,16,24,31])
    def line(axis,path,color,style='-',alpha=1.):
        x=np.array(path)%32; y=np.array(path)//32
        drawn,=axis.plot(x,y,style,color=color,linewidth=1.7,alpha=alpha,zorder=3)
        assert np.array_equal(drawn.get_xdata(),x) and np.array_equal(drawn.get_ydata(),y)
    for pos in sorted(set(selected.values())):
        index=int(ids[pos]); record=accepted[index]; m=int(data['map_id'][index]); w=data['walls'][m]
        s,g=int(data['start'][index]),int(data['goal'][index]); target=data['fields'][data['field_id'][index]]
        oracle=decode(np.maximum(target,0).astype(float),w,s,g)
        assert oracle['status']=='success' and legal_path(oracle['path'],w,s,g) and len(oracle['path'])-1==data['optimal'][index]
        assert s==record['sy']*32+record['sx'] and g==record['gy']*32+record['gx']
        fig,axes=plt.subplots(3,3,figsize=(15,15))
        for axis in axes.flat: base(axis,w,s,g)
        axes.flat[0].set_title('Official map | blue S, gold G')
        line(axes.flat[1],oracle['path'],'#255eb2'); axes.flat[1].set_title(f'BFS optimum L={len(oracle["path"])-1}')
        for axis,mode in zip(list(axes.flat)[2:],MODES):
            result=paths[mode][pos]; path=result['path']
            assert result['record_index']==index and legal_path(path,w,s)
            line(axis,oracle['path'],'#255eb2','--',.2)
            color='#1d9956' if result['status']=='success' else '#9e4eb8' if result['status']=='loop' else '#dd493d'
            line(axis,path,color)
            end=path[-1]; axis.scatter(end%32,end//32,c=color,marker='x',s=45,zorder=6)
            axis.set_title(f'{mode}: success={result["status"]=="success"}\nL={len(path)-1}, optimal={data["optimal"][index]}, {result["status"]}',fontsize=10)
        title=f'{record["map_name"]} | {record["scenario_id"]}\nS=({record["sx"]},{record["sy"]}) G=({record["gx"]},{record["gy"]}) | four-way optimum={data["optimal"][index]} | no graph-search repair'
        fig.suptitle(title,fontsize=11); fig.tight_layout(rect=(0,0,1,.95))
        filename=f'examples/scenario_{index}_paths.png'; fig.savefig(run/filename,dpi=130); plt.close(fig)
        fig,axes=plt.subplots(2,5,figsize=(18,8))
        maps=[h[pos,k]*config['distance_scale'] for k in range(8)]+[scores['D'][pos]*config['distance_scale'],target.astype(float)]
        vmax=max(float(target.max()),float(np.max(maps[8])),1.)
        for axis,array,label in zip(axes.flat,maps,[f'R{k}' for k in range(1,9)]+['D readout','BFS four-way distance']):
            image=np.ma.masked_where((w>0)|(target<0),array)
            obj=axis.imshow(image.reshape(32,32),origin='upper',vmin=0,vmax=vmax,cmap='viridis')
            axis.scatter(s%32,s//32,c='cyan',s=10); axis.scatter(g%32,g//32,c='red',s=15,marker='*')
            axis.set_title(label); axis.set_xticks([]); axis.set_yticks([])
        fig.colorbar(obj,ax=list(axes.flat),label='Distance score in four-way unit steps',shrink=.8)
        fig.suptitle(title,fontsize=10)
        heat=f'examples/scenario_{index}_distance_history.png'; fig.savefig(run/heat,dpi=130); plt.close(fig)
        payload.append({'record_index':index,'map_name':record['map_name'],'scenario_id':record['scenario_id'],
                        'wall_sha256':hashlib.sha256(w.tobytes()).hexdigest(),
                        'start_xy':[record['sx'],record['sy']],'goal_xy':[record['gx'],record['gy']],
                        'oracle_path':oracle['path'],'images':[filename,heat]})
    (run/'visualization_payload.json').write_text(json.dumps({'samples':payload},indent=2),encoding='utf-8')


def report(run,rows,summary,ablations,config):
    ref=read(run/'dataset_reference.json'); audit=read(Path(ref['path'])/'audit.json'); verification=read(run/'verification.json')
    def lookup(mode,group): return next(r for r in rows if r['mode']==mode and r['group']==group)
    def number(row,key):
        value=row.get(key+'_mean'); std=row.get(key+'_std')
        return 'N/A' if value is None else f'{value:.6f} ± {std:.6f}'
    text=['# EXP004 — Moving AI 공식 데이터의 실제 S→G 경로 탐색', '',
          '**Moving AI 공식 MAPF 지도·시나리오에서 추출한 단일 에이전트 경로 탐색 평가**입니다. 공식 단일 에이전트 모델 평가 점수 또는 다중 에이전트 충돌 해결 성능이 아닙니다.', '',
          f'run_id=`{run.name}`, smoke={config["smoke"]}, seed={config["seeds"]}. 코어/헤드/A 각80epoch가 기본이며 이 실행은 {config["epochs_core"]}/{config["epochs_heads"]}/{config["epochs_baselines"]}epoch입니다.', '',
          '## 원본과 파생 과제', '',
          '출처: [공식 MAPF 자료](https://movingai.com/benchmarks/mapf/index.html), [MAPF 안내·인용](https://movingai.com/benchmarks/mapf.html), [파일 형식](https://movingai.com/benchmarks/formats.html), [그리드 자료](https://movingai.com/benchmarks/grids.html). 공식 안내의 Open Data Commons Attribution License를 따르는 연구 자료이며, 출처 인용은 Stern et al., “Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks”, SoCS 2019, pp.151–158입니다.',
          '5개 공식32×32 지도와 각25개 even/25개 random 파일을 원본 ZIP 그대로 보존했습니다. 실제 파일은 version1/9필드 구조였고, 마지막 값은 별도8방향·sqrt(2)·corner-cut 금지 Dijkstra와 대조했습니다. 우리의 최적 길이는 이 값을 쓰지 않고 4방향 단위 비용 BFS와 독립 Dijkstra가 일치하는 값입니다. G/S 지형 문자를 좌표로 해석하지 않습니다. 좌표 원점은 좌상단, x=열/y=행이며 지도 변형·자체 시나리오 생성이 없습니다.',
          f'원본{audit["validation"]["raw_rows"]}행, 파싱 실패{audit["validation"]["parse_failure_rows"]}행. 필터 결과: `{audit["validation"]["counts"]}`. 최소12칸 필터를 완화하지 않았습니다. 원본 비용 대조 최대 오차={audit["validation"]["original_cost_max_abs_difference"]:.3g}. 원본 URL·UTC 다운로드 시각·archive/member SHA-256은 data/movingai_exp004/download_manifest.json, 행별 제외 사유는 준비 데이터의 row_ledger.jsonl에 있습니다.', '',
          '| split | 문제 수 | 실제 거리 min/mean/max | Q1/Q2/Q3/Q4 개수 |', '|---|---:|---|---|']
    for name,d in audit['splits'].items(): text.append(f'| {name} | {d["n"]} | {d["distance_min"]}/{d["distance_mean"]:.2f}/{d["distance_max"]} | {list(d["bands"].values())} |')
    text += ['', 'maze-32-32-4와 random-32-32-20은 학습·검증에서 지도 전체를 제외했습니다. 나머지3개 지도는 목표 지점 단위로80/10/10% 분리했습니다. 거리장 정답은 S와 무관하게 같은 지도·G에서 동일하므로, 새 S만 바꾼 동일 거리장 정답이 학습/테스트에 겹치는 것을 방지한 더 엄격한 동일 지도 평가입니다. S와 G는 모든 모델 입력에 포함됩니다.',
             f'훈련 거리의 nearest 사분위수={audit["difficulty_edges_four_way"]}. Q1=12~21, Q2=22~31, Q3=32~46, Q4≥47이며 테스트로 경계를 정하지 않았습니다. 지도별·구간별 표본 수를 함께 보고합니다. 2개의 미관찰 지도로부터 일반 지도 전체의 일반화를 주장하지 않습니다.', '',
             '## 경로 성공·최적성·오류', '',
             '| 평가 | 모델 | n | 도착 성공률 | 전체 최적 성공률 | 성공한 경로 비용비 | SPL | 합법 경로율 | 정체 실패율 | 거리 MAE |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('seen','unseen'):
        for mode in MODES:
            r=lookup(mode,split)
            text.append(f'| {split} | {mode} | {r["n"]} | {number(r,"success")} | {number(r,"optimal")} | {number(r,"success_cost_ratio")} | {number(r,"spl")} | {number(r,"legal")} | {number(r,"local_minimum")} | {number(r,"distance_mae")} |')
    text += ['', '±는 seed 간 표본 표준편차이며 신뢰구간이나 유의성을 뜻하지 않습니다. 비용비는 성공한 경로만의 조건부 평균이므로 성공률과 전체 최적 성공률·SPL을 별도 제시했습니다. SPL=성공 여부×최적 길이/max(실제 길이,최적 길이)로 실패는0입니다. 최적 경로가 여럿이면 어떤 최단 경로든 인정합니다. 성공한 경로가 없는 조건부 비용비는 N/A입니다.',
             '합법 경로율은 실패한 부분 경로도 포함합니다. 디코더가 벽·범위를 미리 검사하므로 합법성100%는 모델이 벽을 학습했다는 증거가 아닙니다. 모델 점수가 가장 낮은 이웃이 벽인 빈도는 illegal_best_event_rate로 별도 기록합니다. 지도 밖에는 모델 출력 점수가 없으므로 해당 방향을 임의 점수로 순위화하지 않았습니다. 엄격 감소 때문에 루프는 구조적으로 억제되며 이를 학습 성과라고 해석하지 않습니다.', '',
             '## 거리 난이도별 성공률', '', '| 평가 | 거리 구간 | n | A | A_large | A_flops | B | C | D | E |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('seen','unseen'):
        for k in range(1,5):
            group=f'{split}/Q{k}'; d=lookup('D',group)
            text.append(f'| {split} | Q{k} | {d["n"]} | '+' | '.join(number(lookup(m,group),'success') for m in MODES)+' |')
    text += ['', '지도별 전체 지표는 comparison.csv/aggregate.json의 seen/mapN·unseen/mapN에 있습니다. map0~4 순서는 설정 maps 배열과 같습니다. 예측 실제 경로·실패 사유는 *_paths.json에 원본 record_index와 함께 보존합니다.', '',
             '## 연구 질문에 대한 직접 비교', '']
    for split in ('seen','unseen'):
        d=lookup('D',split)['success_mean']; c=lookup('C',split)['success_mean']; b=lookup('B',split)['success_mean']
        text += [f'- {split}: 실제 도착 성공률 D={d:.6f}, C={c:.6f}, B={b:.6f}. C−B={c-b:+.6f}, D−C={d-c:+.6f}, D−A={d-lookup("A",split)["success_mean"]:+.6f}, D−A_large={d-lookup("A_large",split)["success_mean"]:+.6f}, D−A_flops={d-lookup("A_flops",split)["success_mean"]:+.6f}.',
                 '  C/D는 같은 동결 코어와 헤드 초기화·배치를 사용했습니다. C−B에는 추가 최종 손실 학습·파라미터·원본 직접 접근이 포함됩니다. D−C가 전체 기록 추가 이익의 가장 직접적인 비교입니다. 거리장 MAE 감소와 도착 성공은 같은 지표가 아닙니다.']
    text += ['', '## 학습된 D의 기록 교란', '', '| 평가 | 교란 | 도착 성공률 | 정상 D 대비 변화 | 거리 MAE |', '|---|---|---:|---:|---:|']
    for split in ('seen','unseen'):
        normal=lookup('D',split)['success_mean']
        for r in ablations:
            if r['group']==split: text.append(f'| {split} | {r["variant"]} | {number(r,"success")} | {r["success_mean"]-normal:+.6f} | {number(r,"distance_mae")} |')
    text += ['', '교란은 전체0, R8복제, 역순, 고정 무작위 순열 [4,1,7,2,8,5,3,6], 평가 문제 순서를1칸 순환한 다른 공식 시나리오 기록, R1~R8 각각0 대체입니다. 학습된 D는 바꾸지 않았습니다. 분포 밖 입력이므로 저하만으로 과거 기록의 근본적 해결 능력이나 시간 이해를 입증하지 않습니다.', '',
             '## 비용과 학습 조건', '', '| 모델 | 전체 파라미터 | 전체 steps | 코어 포함 학습초 | score GPU ms/batch | 전송+decoder ms/batch | forward MFLOPs/sample | peak allocated MiB |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        records=[r for r in summary if r['mode']==mode]; first=records[0]
        text.append(f'| {mode} | {first["total_system_parameters"]} | {first["system_steps"]} | {np.mean([r["system_train_seconds"] for r in records]):.2f} | {np.mean([r["latency"]["model_only_ms_mean"] for r in records]):.4f} | {np.mean([r["latency"]["model_and_decoder_ms_mean"] for r in records]):.4f} | {first["latency"]["total_flops"]/1e6:.4f} | {np.mean([r["peak_allocated_bytes"] or 0 for r in records])/2**20:.2f} |')
    text += ['', f'AdamW lr={config["learning_rate"]}, weight_decay={config["weight_decay"]}, batch={config["batch_size"]}, FP32, TF32 비활성, deterministic입니다. 마지막 epoch를 고정 사용하고 validation은 관찰용이며 테스트로 모델을 고르지 않았습니다.',
             '정규화 y=d/128, 역변환 d=score×128이며 값은 자르지 않습니다. 모든 모델은 Softplus 양수 점수를 출력합니다. G 연결 성분의 도달 가능 셀만 표본별 평균 MSE에 포함합니다. 벽과 도달 불가 빈칸은 모두 거리 손실에서 제외하되 원본 벽 지도와 거리 mask로 구분합니다. R0는 전부1(128칸 추정)입니다. W/S/G는 매 반복 원본을 유지하며 실제 예측 R을 재입력하고 전체 BPTT로 학습합니다. 모든 단계의 목표는 같은 최종 거리장이며 core loss=0.5×마지막 손실+0.5×8단계 평균(마지막의 총 가중치0.5625)입니다. 헤드/A는 최종 MSE만 사용합니다. BFS 중간 정답을 헤드에 입력하지 않고 G 점수를 강제0으로 바꾸거나 후처리 검색으로 실패를 복구하지 않았습니다.',
             'C/D의 비용에는 동일 코어 사전학습을 각각 포함하며 E에는 코어 비용이 없습니다. A_large는 코어+헤드 파라미터 수 근사 대응, A_flops는 코어8회+헤드1회 선형 forward FLOPs 대응이며 동일 wall-time 학습과는 다릅니다. 선형 multiply-add=2로 계산하고 비선형·cat·역전파·optimizer·CPU 디코더는 FLOPs에 포함하지 않습니다. 캐시 준비와 IO는 별도 기록합니다. 실제 추론 C/D는 코어8회와 헤드1회를 재계산하고 E는 헤드1회만 실행합니다. warm-up·측정 횟수·표준편차·전송 시간·증분 메모리는 summary.json에 있습니다. 8×1024 FP32 기록은 sample당32KiB입니다.', '',
             '## 검증·한계·후속', '',
             f'공식 파일 archive/member SHA-256, 실제9필드·좌표, 전체 BFS/Dijkstra 거리장 {verification["dataset"]["independent_goal_fields"]}개, 적격 문제 전체 oracle 디코더 성공·최적100%, 분할 중복/지도 holdout, 모델 W/S/G 입력·공유 가중치·동결·헤드 초기화, 모든 checkpoint 출력/경로/교란 재현을 확인했습니다. 기존 연구와 기존 공용 데이터 {verification["preserved_files"]}개 파일의 해시가 일치합니다. 시각화도 원본 벽·공식 S/G·oracle 경로·실제 예측 경로를 직접 대조합니다.',
             '거리장 감독은 G가 같으면 S와 무관한 정답입니다. S 입력을 포함해도 이 목적함수가 S 의존 정책 학습을 직접 강제하는 것은 아닙니다. 낮은 전역 MSE가 모든 이웃의 정확한 상대 순서나 G 근처 단조 하강을 보장하지 않습니다. 실패는 수정하지 않고 그대로 보고합니다. 제한된5개지도, holdout2개와 지도 계열 차이, 마지막epoch 고정, MLP의 표현과 학습 설정에 한계가 있습니다. 이전 연결 영역 IoU와 이번 실제 경로 성공률을 합쳐 비교하지 않습니다.',
             '후속 후보는 공식 지도 범위 확장·지도단위 교차 검증, 별도run의학습예산/epoch/거리손실비교입니다. 경로 성공을 올리려고 BFS/A*로 결과를 완성하는 비교는 이번 범위에서 수행하지 않습니다.', '']
    (run/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path); render(parser.parse_args().run)
