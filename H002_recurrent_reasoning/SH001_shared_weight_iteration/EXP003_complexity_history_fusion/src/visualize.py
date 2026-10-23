"""Complexity curves, paired gaps, history interventions and same-maze readout panels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from analyze import read

MODES=['A','A_large','A_flops','B','C','D','E']
BANDS=['Easy','Medium','Hard','Extreme']


def render(run):
    run=Path(run)
    rows,config=read(run/'aggregate.json'),read(run/'config.json')
    gaps,ablations=read(run/'paired_gaps.json'),read(run/'ablation_aggregate.json')
    summary=read(run/'summary.json')
    def lookup(mode,group):
        return next(r for r in rows if r['mode']==mode and r['group']==group)
    for group_axis,labels in [('difficulty',BANDS),('size',[str(x) for x in config['maze_sizes']])]:
        for metric in ('iou','exact'):
            fig,axis=plt.subplots(figsize=(10,5))
            for mode in MODES:
                values=[lookup(mode,group_axis+'/'+label) for label in labels]
                mean=[r.get(metric+'_mean',float('nan')) for r in values]
                std=[r.get(metric+'_std',float('nan')) for r in values]
                axis.errorbar(range(len(labels)),mean,yerr=std,marker='o',capsize=3,label=mode)
            axis.set_xticks(range(len(labels)),labels)
            axis.set(xlabel='BFS eccentricity band' if group_axis=='difficulty' else 'Maze side length',ylabel='Macro IoU' if metric=='iou' else 'Exact full-map fraction',
                     title=f'Valid cells only; epochs={config["epochs_heads"]}; seeds={config["seeds"]}; mean +/- sample SD')
            axis.legend(ncol=4)
            axis.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(run/f'{group_axis}_{metric}.png',dpi=160)
            plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for axis,size in zip(axes,config['maze_sizes']):
        for other in ('A','A_large','A_flops','C','E'):
            selected=[next(r for r in gaps if r['group']==f'size/{size}/difficulty/{b}' and r['comparison']=='D-'+other) for b in BANDS]
            values=[r['mean'] if r['mean'] is not None else np.nan for r in selected]
            std=[r['std'] if r['std'] is not None else np.nan for r in selected]
            axis.errorbar(range(4),values,yerr=std,marker='o',label='D-'+other,capsize=2)
        axis.axhline(0,color='black',linewidth=.8)
        axis.set_xticks(range(4),BANDS,rotation=20)
        axis.set(title=f'{size}x{size}',ylabel='Paired IoU difference')
        axis.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.suptitle('Maze size and distance difficulty are separate axes; absent strata left empty')
    fig.tight_layout()
    fig.savefig(run/'paired_gaps_by_size_difficulty.png',dpi=160)
    plt.close(fig)
    normal=lookup('D','all')['iou_mean']
    variants=[r for r in ablations if r['group']=='all']
    fig,axis=plt.subplots(figsize=(12,5))
    axis.bar(range(len(variants)),[r['iou_mean']-normal for r in variants])
    axis.set_xticks(range(len(variants)),[r['variant'] for r in variants],rotation=55,ha='right')
    axis.set(ylabel='IoU change vs unmodified D',title='Fixed trained D; history interventions can be out of distribution')
    axis.axhline(0,color='black',linewidth=.8)
    fig.tight_layout()
    fig.savefig(run/'history_interventions.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for mode in MODES:
        records=[r for r in summary if r['mode']==mode]
        quality=lookup(mode,'all')['iou_mean']
        costs=[np.mean([r['total_training_seconds_including_core'] for r in records]),
               np.mean([r['latency']['model_only_ms_mean'] for r in records]),
               records[0]['latency']['total_linear_flops_per_sample']/1e6]
        for axis,cost in zip(axes,costs):
            axis.scatter(cost,quality)
            axis.annotate(mode,(cost,quality),fontsize=8)
    for axis,label in zip(axes,['Training seconds incl core (cache separate)','Actual inference ms / batch','Forward linear MFLOPs / sample']):
        axis.set(xlabel=label,ylabel='Test macro IoU')
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(run/'quality_cost.png',dpi=160)
    plt.close(fig)
    panels(run,config)
    report(run,rows,gaps,ablations,summary,config)


def panels(run,config):
    data=np.load(run/'dataset.npz')
    problem,dist,meta,start=[data['test_'+k] for k in ('problem','distances','meta','start')]
    seed=config['seeds'][0]
    history=np.load(run/f'history_seed{seed}_test.npz')['states']
    predictions={mode:np.load(run/f'{mode}_seed{seed}_prediction.npz')['probability'] for mode in MODES}
    metrics={mode:np.load(run/f'{mode}_seed{seed}_sample_metrics.npz')['iou'] for mode in MODES}
    delta=metrics['D']-metrics['C']
    selection={'largest_D_gain_vs_C':int(delta.argmax()),'largest_D_loss_vs_C':int(delta.argmin()),
               'worst_D':int(metrics['D'].argmin()),'longest_distance':int(meta[:,1].argmax())}
    for n in config['maze_sizes']:
        indices=np.flatnonzero(meta[:,0]==n)
        selection[f'size_{n}_first']=int(indices[0])
    indices=sorted(set(selection.values()))
    (run/'example_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    palette=ListedColormap(['#efefef','#343a40','#2ca96a','#df4545','#f0b63f','#aaaacc'])
    def draw(axis,index,pred,title):
        p=problem[index]
        valid=p[:256].astype(bool)
        wall=p[256:512].astype(bool)
        truth=dist[index]>=0
        colors=np.zeros(256,dtype=int)
        colors[wall&valid]=1
        if pred is not None:
            pred=pred>=.5
            colors[pred&truth]=2
            colors[pred&~truth&valid]=3
            colors[~pred&truth]=4
        colors[~valid]=5
        axis.imshow(colors.reshape(16,16),cmap=palette,vmin=0,vmax=5,interpolation='nearest')
        gr,gc=divmod(int(p[512:].argmax()),16)
        sr,sc=divmod(int(start[index]),16)
        axis.scatter(gc,gr,s=12,c='#2389ff',marker='o',edgecolors='black',linewidths=.3)
        axis.scatter(sc,sr,s=20,c='#e34cce',marker='*',edgecolors='black',linewidths=.3)
        axis.set_title(title,fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    for name,kind in [('core_history','history'),('final_outputs','outputs')]:
        columns=10 if kind=='history' else 9
        fig,axes=plt.subplots(len(indices),columns,figsize=(columns*1.7,len(indices)*1.8+1),squeeze=False)
        for row,index in enumerate(indices):
            draw(axes[row,0],index,None,f'#{index} n={int(meta[index,0])} d={int(meta[index,1])}')
            draw(axes[row,1],index,(dist[index]>=0).astype(float),'BFS final')
            if kind=='history':
                for k in range(8):
                    draw(axes[row,k+2],index,history[index,k],f'R{k+1}')
            else:
                for col,mode in enumerate(MODES,2):
                    draw(axes[row,col],index,predictions[mode][index],mode)
        patches=[Patch(color=palette(i),label=label) for i,label in enumerate(['unreached open','wall','correct discovery','false positive','missed reachable','invalid padding'])]
        fig.legend(handles=patches,loc='lower center',ncol=6,fontsize=8)
        fig.suptitle(f'{name}; seed={seed}; blue=G, magenta=S; identical indices across panels')
        fig.tight_layout(rect=(0,.035,1,.98))
        fig.savefig(run/(name+'.png'),dpi=150)
        plt.close(fig)
    # Cell-level corrections made by D relative to C and B; purple = introduced error.
    fig,axes=plt.subplots(len(indices),3,figsize=(8,len(indices)*2),squeeze=False)
    change_palette=ListedColormap(['#eeeeee','#28a66a','#a452bd','#343a40','#aaaacc'])
    for row,index in enumerate(indices):
        truth=dist[index]>=0
        valid=problem[index,:256].astype(bool)
        for col,other in enumerate(['B','C','E']):
            before=predictions[other][index]>=.5
            after=predictions['D'][index]>=.5
            values=np.zeros(256,dtype=int)
            values[(before!=truth)&(after==truth)&valid]=1
            values[(before==truth)&(after!=truth)&valid]=2
            values[(problem[index,256:512]==1)&valid&(values==0)]=3
            values[~valid]=4
            axes[row,col].imshow(values.reshape(16,16),cmap=change_palette,vmin=0,vmax=4)
            axes[row,col].set_title(f'#{index}: D vs {other}; green fixed, purple broken',fontsize=8)
            axes[row,col].axis('off')
    fig.tight_layout()
    fig.savefig(run/'fusion_corrections.png',dpi=150)
    plt.close(fig)


def report(run,rows,gaps,ablations,summary,config):
    def lookup(mode,group='all'):
        return next(r for r in rows if r['mode']==mode and r['group']==group)
    def number(mode,key='iou',group='all'):
        r=lookup(mode,group)
        return f'{r[key+"_mean"]:.5f} ± {r[key+"_std"]:.5f}' if r['n'] else 'N/A (n=0)'
    audit=read(run/'dataset_audit.json')
    verification=read(run/'verification.json')
    text=['# EXP003 — 복잡도와 기록 통합 실제 실행 보고', '',f'run_id: `{run.name}`, smoke={config["smoke"]}, seeds={config["seeds"]}.',
          f'코어 {config["epochs_core"]}epoch, 헤드 {config["epochs_heads"]}epoch, A들 {config["epochs_baselines"]}epoch; AdamW lr={config["learning_rate"]}, batch={config["batch_size"]}, FP32. 미로 크기별 train/val/test={config["sizes_per_maze_size"]}.', '',
          '## 전체 최종 영역 품질', '',
          '| 방식 | IoU | F1 | BCE | 완전 해결률 | S 정확도 | 누락률 | 분리 영역 오탐률 | 고확신 오답 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        text.append('| '+mode+' | '+' | '.join(number(mode,key) for key in ('iou','f1','bce','exact','start_accuracy','fnr','disconnected_fpr','confident_wrong'))+' |')
    text+=['', '유효 셀에 한정한 샘플별 지표를 평균했습니다. ±는 3seed 표본 표준편차(smoke는 0)이며 신뢰구간이나 통계적 유의성을 뜻하지 않습니다. BCE는 헤드/단일 모델에서 logits 기반의 안정적 계산, B에서 확률 BCE입니다. S는 모델에 입력되지 않습니다.', '',
           '## 여섯 연구 질문', '']
    for label in BANDS:
        group='difficulty/'+label
        if lookup('D',group)['n']:
            da=lookup('D',group)['iou_mean']-lookup('A',group)['iou_mean']
            dc=lookup('D',group)['iou_mean']-lookup('C',group)['iou_mean']
            de=lookup('D',group)['iou_mean']-lookup('E',group)['iou_mean']
            text.append(f'- 질문 1·6, {label} (n={lookup("D",group)["n"]}): D−A={da:+.5f}, D−C={dc:+.5f}, D−E={de:+.5f}. 크기별 교차 표를 함께 확인해야 하며 난이도 간 표본 구성은 다릅니다.')
        else:
            text.append(f'- 질문 1·6, {label}: 생성된 테스트 표본이 없어 비교하지 않습니다.')
    text.append(f'- 질문 2: 출력 MLP가 있는 C−B의 전체 IoU 차이={lookup("C")["iou_mean"]-lookup("B")["iou_mean"]:+.5f}. 이 차이에는 원본 문제 직접 접근·추가 파라미터·추가 최종 목표 학습이 함께 포함됩니다.')
    text.append(f'- 질문 3: 실제 기록 D−마지막 복제 C의 전체 IoU 차이={lookup("D")["iou_mean"]-lookup("C")["iou_mean"]:+.5f}. 동일한 동결 코어와 헤드 초기화·배치 순서에서의 비교입니다. C의 중복 슬롯은 입력 가중치 합으로 표현할 수 있어 명목 파라미터 수와 유효 기능 차원은 구분해야 합니다.')
    text.append(f'- 질문 4: D−원본 전용 E의 전체 IoU 차이={lookup("D")["iou_mean"]-lookup("E")["iou_mean"]:+.5f}. 아래 학습된 D의 교란 검사와 함께 해석합니다. E는 명목 헤드 구조는 같지만 0 슬롯의 가중치가 손실에 연결되지 않아 기본 A와 같은 함수 형태입니다. 초기화·최적화까지 동일한 실험은 아닙니다.')
    text.append(f'- 질문 5: D−파라미터 대응 A_large={lookup("D")["iou_mean"]-lookup("A_large")["iou_mean"]:+.5f}, D−추론 선형 FLOPs 대응 A_flops={lookup("D")["iou_mean"]-lookup("A_flops")["iou_mean"]:+.5f}. 동일 wall-time 학습 비교는 수행하지 않았습니다. 아래 비용 표를 함께 보아야 하며 단순 효율 우위를 주장하지 않습니다.')
    text+=['', '## 크기 × BFS 거리별 품질', '', '| 크기 | 난이도 | n | A | A_large | A_flops | B | C | D | E |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for size in config['maze_sizes']:
        for band in BANDS:
            group=f'size/{size}/difficulty/{band}'
            text.append(f'| {size} | {band} | {lookup("D",group)["n"]} | '+' | '.join(number(m,group=group) for m in MODES)+' |')
    text+=['', 'Easy=거리4~8, Medium=9~16, Hard=17~32, Extreme≥33입니다. 기준은 G 연결 성분의 최대 BFS 거리입니다. 없는 구간은 조건을 완화하거나 채워 넣지 않았습니다. 분기·막다른 길·성분 수·우회 비율·생성 계열별 모든 결과는 comparison.csv와 aggregate.json에 있습니다.', '',
           '## 원래 D의 기록 교란', '', '| 교란 | IoU | 정상 D 대비 IoU 변화 |', '|---|---:|---:|']
    normal=lookup('D')['iou_mean']
    for r in ablations:
        if r['group']=='all':
            text.append(f'| {r["variant"]} | {r["iou_mean"]:.5f} ± {r["iou_std"]:.5f} | {r["iou_mean"]-normal:+.5f} |')
    text+=['', 'repeat_last=R8 복제, zero_all=기록 전부 제거, reverse=시간 역순, shuffle=고정 순열 [4,1,7,2,8,5,3,6], other_sample_all=테스트 순서를 한 칸 순환한 다른 샘플 기록, other_sample_R4=다른 샘플 R4만 대체, zero_Rk=해당 슬롯만 0입니다. 코어와 D를 재학습하지 않았습니다. 교란은 분포 밖 입력일 수 있어 성능 변화가 인간이 이해하는 기록 활용 방식의 인과적 증거는 아닙니다.', '',
           '## 계산 비용과 공정성', '', '| 방식 | 전체 시스템 파라미터 | 이번 단계 학습 파라미터 | core 포함 steps | core 포함 학습초 | 코어/헤드 train 호출 | 모델 추론 ms/batch | MFLOPs/sample | peak allocated MiB |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        records=[r for r in summary if r['mode']==mode]
        avg=lambda key:float(np.mean([r[key] for r in records]))
        latency=np.mean([r['latency']['model_only_ms_mean'] for r in records])
        text.append(f'| {mode} | {records[0]["total_system_parameters"]} | {records[0]["trainable_parameters"]} | {records[0]["total_training_steps_including_core"]} | {avg("total_training_seconds_including_core"):.2f} | {records[0]["training_core_calls"]}/{records[0]["train_head_calls"]} | {latency:.4f} | {records[0]["latency"]["total_linear_flops_per_sample"]/1e6:.4f} | {avg("train_peak_allocated_bytes")/2**20 if records[0]["train_peak_allocated_bytes"] else 0:.2f} |')
    text+=['', '코어=328,192, 헤드=786,944, 합계=1,115,136개입니다. A_large=1,115,456개로 약 0.0287% 차이입니다. A_flops의 폭은 3,328이며 C/D의 8회 코어+1회 헤드와 선형층 forward FLOPs가 정확히 같습니다. 활성화·마스킹·cat·메모리 접근·역전파·optimizer는 FLOPs 추정에서 제외되어 실측 시간 일치나 총 학습 연산량 일치를 뜻하지 않습니다.',
           'C/D의 총 비용에는 같은 코어 사전학습 비용을 각각 포함해 독립 시스템으로 비교합니다. 실제 실행에서는 코어를 seed별 한 번만 훈련해 공유했으므로 시스템별 합계를 더하면 실제 사용 시간과 다릅니다. E는 훈련·추론에 코어가 필요 없으므로 코어 비용을 부과하지 않습니다. 학습 시간은 검증·gradient 기록을 포함하고 캐시 생성·압축·테스트·저장·시각화는 별도입니다. 캐시 비용/크기/해시는 cache_seed*.json에 있습니다.',
           f'지연은 실제 전체 시스템을 다시 계산해 측정했습니다. C/D는 캐시를 사용하지 않고 코어8회+헤드1회, E는 헤드1회입니다. warm-up={config["latency_warmup"]}, 측정={config["latency_repeats"]}, batch={min(config["batch_size"],sum([config["sizes_per_maze_size"][2] for _ in config["maze_sizes"]]))}. CUDA 동기화 wall time, 평균·표준편차와 CPU↔GPU 전송 포함 시간은 summary.json에 보존합니다. C/D의 8개 FP32 기록은 sample당 8,192byte, batch128에서 1MiB입니다. peak allocated는 상주 데이터/캐시가 포함될 수 있으며 증분 peak도 별도 기록합니다.', '',
           '## 데이터 감사·무결성·재현', '',
           '무작위 장애물과 DFS 통로의 무작위 간선 절단을 같은 수로 생성합니다. G는 열린 칸에서 선택하며 eccentricity≥4, 분리된 열린 셀≥2 및 열린 셀의 최소8%를 요구합니다. 각 크기에서 고유 벽 지도를 먼저 셔플해 분리하며 회전·반사한 지도의 split 간 중복도 없습니다. 캔버스 배치 위치와 방향은 변하고 W 패딩은 0, V 패딩은 0, 거리 패딩은 -1입니다. 코어 R은 V로 마스킹하고 loss/IoU/F1은 유효 셀만 사용합니다. 단순한 벽 밀도가 난이도 전체를 뜻하지 않습니다.',
           f'데이터 재생성과 중복 검사, 독립 flood-fill {verification["independent_flood_cases"]}개 정답 검사, 공유 코어8회·모든 단계 gradient·문맥/마스크 보존, teacher forcing 없음, 동결 코어 불변, C/D/E 대응 초기화·배치 검사, {len(verification["checkpoints"])}개 시스템 체크포인트 출력 및 모든 D 교란 재현, 독립 FP64 BCE·IoU 검증을 통과했습니다. 기존 H001·EXP001·EXP002·공용 데이터 {verification["preserved_files"]}개 파일 SHA-256과 원본 소스 스냅샷이 일치합니다.',
           'BFS 중간 정답은 코어 손실 계산에만 사용됩니다. 헤드 기록은 동결 코어에서 생성한 캐시이며 전체 train/val/test를 모델 출력과 다시 대조했습니다. 헤드 출력은 코어로 되돌아가지 않습니다. NaN/Inf를 검사하고 실패 시 자동 수정·삭제하지 않습니다. checkpoints/<run_id>/에 모델/optimizer를, results/<run_id>/에 데이터·기록·예측·교란·수치·그림·source/·해시를 보존하며 기존 ignore 정책을 따릅니다.', '',
           '## 한계와 후속 후보', '',
           '관측된 지표 차이만으로 학습 부족·기록 활용·구조적 우위의 원인을 확정하지 않습니다. 생성기 계열·복잡도 구간의 표본 구성과 편향이 존재합니다. 완전 관측 문제에서 높은 성능이 반복 필수성이나 자율적 사고를 뜻하지 않습니다. 학습 8기록에 고정된 출력 헤드이며 16/32개 전체 기록 또는 무한 메모리를 평가하지 않았습니다. End-to-end 코어 재학습은 이번 원본 조건에 포함하지 않았습니다.',
           '후속 후보: 독립 생성기의 크기/거리별 holdout, core 품질·학습시간 예산 대응, 기록 슬롯의 정보량과 함수 차원을 맞춘 대조군, 별도 run의 C/D end-to-end 학습, 길이8 윈도우와 가변길이 헤드의 명시적 분리 비교. 원래 실행을 보존한 뒤 수행해야 합니다.', '']
    (run/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    render(parser.parse_args().run)
