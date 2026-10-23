"""Shared sample panels, full 1..32 curves, objective-aware Korean report."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from evaluate import scores

LABELS={'A':'A single-step final','B':'B pure recurrence final8',
        'C':'C preserved context final8','D':'D preserved context stepwise'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def aggregate(summary,selected):
    rows=[]
    keys=['final_bce','macro_iou','macro_f1','step_bce','step_macro_iou','step_pixel_error',
          'start_accuracy','unreachable_open_fpr','reachable_fnr','entropy_nats',
          'high_confidence_wrong_fraction_all_cells','r_change_mean','period2_fraction']
    for mode in LABELS:
        runs=[s for s in summary if s['mode']==mode]
        for k in selected:
            row={'mode':mode,'step':k,'n_seeds':len(runs)}
            for key in keys:
                values=[s['metrics'][k-1][key] for s in runs]
                row[key+'_mean']=float(np.mean(values))
                row[key+'_std']=float(np.std(values,ddof=1 if len(values)>1 else 0))
            rows.append(row)
    return rows


def select_examples(outputs,data):
    final=data['test_distances']>=0
    improvements=[]
    failures=[]
    cycles=[]
    fixed_wrong=[]
    not_converged=[]
    for mode,states in outputs.items():
        predictions=states[:,:,128:]>=.5
        ious=np.array([((p&final).sum(1)/np.maximum((p|final).sum(1),1)) for p in predictions])
        delta=ious[7]-ious[0]
        improvements.append((float(delta.max()),mode,int(delta.argmax())))
        failures.append((float(ious[7].min()),mode,int(ious[7].argmin())))
        r=states[:,:,128:]
        consecutive=np.abs(r[-1]-r[-2]).mean(1)
        two_step=np.abs(r[-1]-r[-3]).mean(1)
        mask=(two_step<1e-5)&(consecutive>1e-4)
        if mask.any():
            index=int(np.flatnonzero(mask)[0])
            cycles.append((float(consecutive[index]),mode,index))
        mask=(consecutive<1e-5)&(ious[-1]<.5)
        if mask.any():
            index=int(np.flatnonzero(mask)[0])
            fixed_wrong.append((float(ious[-1,index]),mode,index))
        if consecutive.max()>1e-4:
            not_converged.append((float(consecutive.max()),mode,int(consecutive.argmax())))
    def record(item,meaning):
        return {'value':item[0],'mode':item[1],'index':item[2],'criterion':meaning} if item else None
    best=max(improvements)
    selection={
        'improvement':record(best if best[0]>0 else None,'largest IoU(k8)-IoU(k1), first seed across all modes'),
        'failure':record(min(failures),'lowest final-target IoU(k8), first seed across all modes'),
        'period2':record(max(cycles) if cycles else None,'at k32: R two-step MAE<1e-5 and one-step MAE>1e-4'),
        'wrong_fixed_point':record(min(fixed_wrong) if fixed_wrong else None,'at k32: R change<1e-5 and IoU<0.5'),
        'not_converged':record(max(not_converged) if not_converged else None,'largest k32 R change if >1e-4'),
        'long_path':{'index':int(data['test_distances'].max(1).argmax()),
                     'criterion':'maximum BFS eccentricity in test split'}}
    indices=sorted({r['index'] for r in selection.values() if r is not None})
    return selection,indices


def panels(run,outputs,data,config):
    selection,indices=select_examples(outputs,data)
    (run/'example_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    palette=ListedColormap(['#f5f5f5','#343a40','#28a66a','#e34a4a','#f4b942'])
    final=data['test_distances']>=0
    def draw(axis,index,prediction,title):
        walls=data['test_walls'][index].astype(bool)
        colors=np.zeros(64,dtype=int)
        colors[walls]=1
        if prediction is not None:
            colors[prediction & final[index] & ~walls]=2
            colors[prediction & ~final[index]]=3
            colors[~prediction & final[index]]=4
            # A predicted wall is red, making illegal detections visible.
        axis.imshow(colors.reshape(8,8),cmap=palette,vmin=0,vmax=4,interpolation='nearest')
        gr,gc=divmod(int(data['test_goal'][index].argmax()),8)
        sr,sc=divmod(int(data['test_start'][index]),8)
        axis.scatter(gc,gr,c='#2374e1',s=30,marker='o',edgecolors='black',linewidths=.5)
        axis.scatter(sc,sr,c='#d14fe0',s=50,marker='*',edgecolors='black',linewidths=.5)
        axis.set_title(title,fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    for mode,states in outputs.items():
        fig,axes=plt.subplots(len(indices),9,figsize=(16,len(indices)*1.7+1),squeeze=False)
        for row,index in enumerate(indices):
            d=data['test_distances'][index]
            draw(axes[row,0],index,None,f'#{index} Maze')
            draw(axes[row,1],index,final[index],'BFS final')
            draw(axes[row,2],index,(d>=0)&(d<=8),'BFS R8 vs final')
            for col,k in enumerate(config['selected_steps'],3):
                draw(axes[row,col],index,states[k-1,index,128:]>=.5,f'{mode} k={k}')
        legend=[Patch(color=palette(i),label=t) for i,t in enumerate(['unreached open','wall','true positive','false positive (incl walls)','missed reachable'])]
        fig.legend(handles=legend,loc='lower center',ncol=5,fontsize=8)
        fig.suptitle(f'{LABELS[mode]} | seed={config["seeds"][0]} | blue circle G, magenta star S; same sample indices')
        fig.tight_layout(rect=(0,.04,1,.97))
        fig.savefig(run/f'examples_{mode}.png',dpi=150)
        plt.close(fig)


def render(run):
    run=Path(run)
    summary,config=load(run/'summary.json'),load(run/'config.json')
    dataset_audit,verification=load(run/'dataset_audit.json'),load(run/'verification.json')
    data=dict(np.load(run/'dataset.npz'))
    rows=aggregate(summary,config['selected_steps'])
    (run/'aggregate.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (run/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    conditions=f'8x8 maze | 192-256-192 shared MLP | epochs={config["epochs"]} | seeds={config["seeds"]}'
    plots=[('macro_iou','Final-target macro IoU'),('macro_f1','Final-target macro F1'),
           ('final_bce','Final-target BCE (nats)'),('step_macro_iou','BFS R_k macro IoU'),
           ('step_pixel_error','BFS R_k pixel error fraction'),('entropy_nats','R entropy (nats / cell)'),
           ('r_change_mean','Mean absolute R state change'),('state_change_mean','Mean absolute effective full state change'),
           ('high_confidence_wrong_fraction_all_cells','High-confidence wrong cells / all cells'),
           ('effective_input_w_mae','Effective wall-input MAE'),('effective_input_g_mae','Effective goal-input MAE')]
    for key,label in plots:
        fig,axis=plt.subplots(figsize=(9,5))
        for mode in LABELS:
            runs=[s for s in summary if s['mode']==mode]
            values=np.array([[r[key] for r in s['metrics']] for s in runs])
            mean,std=values.mean(0),values.std(0,ddof=1 if len(runs)>1 else 0)
            x=np.arange(1,config['eval_steps']+1)
            line,=axis.plot(x,mean,label=LABELS[mode])
            axis.fill_between(x,mean-std,mean+std,color=line.get_color(),alpha=.15)
            for v in values:
                axis.plot(x,v,color=line.get_color(),alpha=.2,linewidth=.6)
        if key=='macro_iou':
            axis.axhline(dataset_audit['test']['all_open_baseline']['macro_iou'],color='black',linestyle='--',label='all open cells baseline')
        axis.axvline(8,color='gray',linestyle=':',linewidth=1)
        axis.set(xlabel='Inference iterations',ylabel=label,title=conditions)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(run/(key+'.png'),dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for mode in LABELS:
        runs=[s for s in summary if s['mode']==mode]
        for axis,key in zip(axes,('model_only_ms_mean','end_to_end_ms_mean')):
            values=np.array([[t[key] for t in s['latency']] for s in runs])
            axis.plot(np.arange(1,config['eval_steps']+1),values.mean(0),label=LABELS[mode])
            axis.set(xlabel='Iterations',ylabel='ms / batch',title=key)
            axis.legend(fontsize=7)
    fig.suptitle(f'{config["device"]}; batch={min(config["batch_size"],config["sizes"][2])}; warmup={config["latency_warmup"]}; repeats={config["latency_repeats"]}')
    fig.tight_layout()
    fig.savefig(run/'latency.png',dpi=160)
    plt.close(fig)
    seed=config['seeds'][0]
    outputs={m:np.load(run/f'{m}_seed{seed}_states.npz')['states'] for m in LABELS}
    panels(run,outputs,data,config)
    report(run,summary,rows,config,dataset_audit,verification)


def report(run,summary,rows,config,data_audit,verification):
    def row(mode,k):
        return next(r for r in rows if r['mode']==mode and r['step']==k)
    def val(mode,k,key):
        r=row(mode,k)
        return f'{r[key+"_mean"]:.5f} ± {r[key+"_std"]:.5f}'
    text=['# EXP002 — 미로 반복 추론 실제 실행 보고', '',f'실행 ID: `{run.name}`. Smoke: `{config["smoke"]}`.',
          f'3개 seed 기본 설정은 42/123/2026, {config["epochs"]}epoch, AdamW lr={config["learning_rate"]}, FP32입니다. 이 실행의 seed는 {config["seeds"]}입니다.', '',
          '## 목표를 구분한 측정', '',
          'A는 1회에 R_final, B/C는 8회에 R_final을 학습했습니다. D는 각 k의 R_k를 감독했습니다. 아래 최종 영역 품질과 단계별 알고리즘 재현은 서로 다른 지표입니다. IoU/F1은 샘플별 값의 평균이며 ±는 seed 간 표본 표준편차입니다.', '',
          '| 방식 | 최종 IoU 1회 | 최종 IoU 8회 | 최종 IoU 32회 | R8 IoU 8회 | 시작점 정확도 8회 |',
          '|---|---:|---:|---:|---:|---:|']
    for mode in LABELS:
        text.append(f'| {mode} | {val(mode,1,"macro_iou")} | {val(mode,8,"macro_iou")} | {val(mode,32,"macro_iou")} | {val(mode,8,"step_macro_iou")} | {val(mode,8,"start_accuracy")} |')
    text += ['', '| 방식 | 8회 BCE(final) | 8회 F1(final) | 분리된 빈칸 오탐률 8회 | 도달 영역 누락률 8회 |', '|---|---:|---:|---:|---:|']
    for mode in LABELS:
        text.append(f'| {mode} | {val(mode,8,"final_bce")} | {val(mode,8,"macro_f1")} | {val(mode,8,"unreachable_open_fpr")} | {val(mode,8,"reachable_fnr")} |')
    text += ['', '## 연구 질문별 답변', '']
    for mode in LABELS:
        d8=row(mode,8)['macro_iou_mean']-row(mode,1)['macro_iou_mean']
        d32=row(mode,32)['macro_iou_mean']-row(mode,8)['macro_iou_mean']
        text.append(f'- 질문 1·4, {mode}: 최종 영역 IoU의 평균 1→8 변화={d8:+.5f}, 8→32 변화={d32:+.5f}. 이는 선택 지점의 변화이며 단계별 단조 개선이나 샘플 전부의 개선을 뜻하지 않습니다.')
    difference=row('C',8)['macro_iou_mean']-row('B',8)['macro_iou_mean']
    text += [f'- 질문 2: C−B의 8회 최종 IoU 평균 차이={difference:+.5f}. 원본 입력 보존을 포함한 구조 차이의 관측이며, 유효한 출력 헤드의 역할도 다르므로 입력 보존만의 인과 효과로 단정하지 않습니다.',
             f'- 질문 3: D의 8회 R8 IoU={row("D",8)["step_macro_iou_mean"]:.5f}, 32회 R32 IoU={row("D",32)["step_macro_iou_mean"]:.5f}. 중간 목표를 어느 정도 재현하는지와 최종 도달 영역을 완전히 계산했는지는 구분해야 합니다. 정확한 전파 학습 여부는 아래 seed별 R_k 오류·영역 확장·누락·장기 안정성을 함께 판단하십시오.',
             '- 질문 5: 아래 엔트로피와 고확신 오답, 상태 변화, 고정점/2주기 수치를 정답 지표와 함께 제시합니다. 낮은 엔트로피나 작은 상태 변화만으로 올바른 추론이라고 해석하지 않습니다.',
             '- 질문 6: optimizer step은 같지만 B/C/D의 학습 코어 호출은 A의 8배입니다. 추론 32회 선형층 FLOPs는 1회의 32배입니다. 시간은 아래 실측을 사용하며 동일 연산 예산의 우위는 검증하지 않았습니다.', '',
             '## Seed별 결과와 비용', '',
             '| 모델 | IoU1 | IoU8 | IoU32 | R8 IoU8 | R32 IoU32 | 학습 초 | steps | 코어 호출 | peak MiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        a,b,c=[s['metrics'][k-1] for k in (1,8,32)]
        memory=s['peak_gpu_allocated_bytes']
        text.append(f'| {s["name"]} | {a["macro_iou"]:.5f} | {b["macro_iou"]:.5f} | {c["macro_iou"]:.5f} | {b["step_macro_iou"]:.5f} | {c["step_macro_iou"]:.5f} | {s["train_seconds_including_validation"]:.2f} | {s["optimizer_steps"]} | {s["train_mapping_calls"]} | {memory/2**20 if memory else 0:.2f} |')
    text += ['', '## 확신·수렴·정보 왜곡 실측', '',
             '| 모델 | entropy 1→32 | 고확신 오답 비율 1→32 | R 변화32 | 고정점 비율32 | 2주기 최대 비율 | 입력 W/G MAE32 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        a,c=s['metrics'][0],s['metrics'][-1]
        text.append(f'| {s["name"]} | {a["entropy_nats"]:.5f}→{c["entropy_nats"]:.5f} | {a["high_confidence_wrong_fraction_all_cells"]:.5f}→{c["high_confidence_wrong_fraction_all_cells"]:.5f} | {c["r_change_mean"]:.3g} | {c["fixed_point_fraction"]:.5f} | {max(m["period2_fraction"] for m in s["metrics"]):.5f} | {c["effective_input_w_mae"]:.5f}/{c["effective_input_g_mae"]:.5f} |')
    text += ['', '고정점/2주기는 실제로 다음 단계에 전달되는 유효 상태로 판정합니다. C/D의 W/G는 원본으로 고정하며, 사용하지 않는 raw W/G 출력은 보존 정보로 해석하지 않습니다. 고정점: 직전 평균 절대 변화<1e-5. 2주기: 2단계 전 변화<1e-5 및 직전 변화>1e-4. 기준에 맞지 않는 다른 주기·느린 변화가 없음을 증명하지 않습니다. 고확신 오답은 p≥0.9 또는 p≤0.1이며 잘못 분류한 셀 / 전체 셀입니다.', '',
             '## 실제 영역 확장', '', '| 모델 | k8 영역 크기 | k8 신규 올바른/잘못된 영역 | k32 신규 올바른/잘못된 영역 | k32 올바른 영역 제거 |', '|---|---:|---:|---:|---:|']
    for s in summary:
        b,c=s['metrics'][7],s['metrics'][-1]
        text.append(f'| {s["name"]} | {b["predicted_area_mean"]:.3f} | {b["correct_new_area_mean"]:.3f}/{b["incorrect_new_area_mean"]:.3f} | {c["correct_new_area_mean"]:.3f}/{c["incorrect_new_area_mean"]:.3f} | {c["correct_area_removed_mean"]:.3f} |')
    text += ['', '신규 영역/제거는 바로 직전 단계의 0.5 이진화 상태와 비교한 샘플당 평균 셀 수입니다. 초기 상태는 G입니다. 올바른/잘못된 확장은 R_final을 기준으로 하므로 해당 단계의 R_k보다 앞선 올바른 셀도 포함할 수 있습니다.', '',
             '## 미로 생성 및 편향 감사', '',
             '장벽으로 분리한 무작위 지도와 분리된 DFS 통로 지도를 같은 수로 생성합니다. 최소 BFS eccentricity 8, 도달 불가능한 빈칸 최소 6개 및 전체 빈칸의 20% 이상을 요구합니다. 지도 풀을 먼저 셔플하고 분할합니다. 벽 지도가 회전·반사로 같아도 split 전체에서 중복을 제거합니다. 각 split의 S는 도달 가능/불가능 50:50이며 가능한 S는 긴 경로에서 선택합니다. S는 모델 입력에 포함되지 않습니다.', '',
             '| split | n | 평균 벽 밀도 | 평균 eccentricity | 최대 eccentricity | eccentricity>16 비율 | all-open IoU | all-open 픽셀오류 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('train','val','test'):
        d=data_audit[split]
        text.append(f'| {split} | {d["n"]} | {d["wall_density"]["mean"]:.4f} | {d["eccentricity"]["mean"]:.3f} | {d["eccentricity"]["max"]:.0f} | {d["fraction_eccentricity_gt16"]:.4f} | {d["all_open_baseline"]["macro_iou"]:.4f} | {d["all_open_baseline"]["pixel_error"]:.4f} |')
    text += ['', 'all-open은 모든 통행 가능 셀을 칠하는 기준선입니다. 이 기준선의 분리된 빈칸 오탐률은 1, 균형 S 정확도는 0.5입니다. 벽이 많으면 all-zero의 픽셀 오류가 낮아 보일 수 있으므로 IoU/F1·누락률·S 정확도를 함께 확인합니다. 생성 제약은 일반 미로 분포를 대표하지 않으며 경로 끝점에서 G를 선택하고 도달 가능한 S를 멀리 선택하는 편향이 있습니다.', '',
             '## 추론 지연과 FLOPs', '', '| 방식 | 반복 | 모델 ms/batch | 전송 포함 ms/batch | 선형층 FLOPs/sample |', '|---|---:|---:|---:|---:|']
    for mode in LABELS:
        runs=[s for s in summary if s['mode']==mode]
        for k in config['selected_steps']:
            timings=[s['latency'][k-1] for s in runs]
            text.append(f'| {mode} | {k} | {np.mean([t["model_only_ms_mean"] for t in timings]):.4f} | {np.mean([t["end_to_end_ms_mean"] for t in timings]):.4f} | {timings[0]["linear_forward_flops_per_sample"]} |')
    text += ['', f'장치={config["device"]}, batch={min(config["batch_size"],config["sizes"][2])}, warm-up={config["latency_warmup"]}, 측정 반복={config["latency_repeats"]}. CUDA 동기화를 포함한 wall time입니다. 전송 포함은 CPU W/G→장치, 모델 실행, 마지막 192차원 출력→CPU입니다. 원시 측정 평균·표준편차는 *_latency.json에 저장합니다. FLOPs는 multiply-add=2로 선형층만 추정하며 활성화·bias·cat·역전파·optimizer는 제외합니다. 학습 시간은 validation·기록을 포함하며 테스트·저장·그래프는 제외합니다. peak는 allocated 메모리이며 드라이버 총 점유량이 아닙니다.', '',
             '## 검증 및 산출물', '',
             f'체크포인트 {len(verification["checkpoints"])}개 복원 출력 최대 차이 0, torch 독립 BCE·IoU·픽셀오류 검증 통과. 수작업 BFS·대각선 금지·Manhattan 거리 검사 및 독립 동기식 flood-fill로 {verification["independent_flood_oracle_mazes"]}개 미로의 단계별/최종 정답을 검증했습니다. 기존 H001·EXP001·공용 데이터 {verification["preserved_files"]}개 파일 SHA-256이 모두 일치했습니다. 같은 seed 초기 가중치 및 배치 순서 해시 일치, 모든 중간 R gradient·추론 가중치 불변 검증도 통과했습니다.',
             '모든 방식의 명목 파라미터는 98,752개입니다. A/C/D에서 손실에 연결되는 파라미터는 65,856개이며 출력 W/G 헤드에는 손실 gradient가 없습니다. AdamW weight decay로 사용하지 않는 파라미터 값이 변할 수는 있습니다. B는 앞선 단계의 W/G 출력이 다음 계산에 사용되므로 모든 헤드에 역할이 있습니다.',
             'config.json, environment.json, generation.json, dataset_audit.json, dataset.npz(벽·G·S·전체 BFS 거리), summary.json, *_history.json, *_states.npz(전체 192차원), *_metrics.json, *_latency.json, aggregate.json, comparison.csv, source/ 및 해시, preservation_before.json, verification.json, 그림, REPORT.md를 보존합니다. 체크포인트는 checkpoints/<run_id>/에 optimizer와 함께 보존하며 기존 ignore 정책을 따릅니다.', '',
             '## 해석 한계와 다음 실험', '',
             'D의 단계 목표 재현은 중간 과정 지도학습의 결과입니다. 자율적 사고나 스스로 계산 전략을 발견했다는 증거가 아닙니다. 지표 변화만으로 학습 부족·정보 손실·목표 부적합성의 인과 원인을 확정하지 않습니다. 학습 8회를 넘은 32회는 같은 크기 미로에서의 길이 외삽이며 더 큰 미로에 대한 일반화가 아닙니다. 단일 지표로 네 방식을 순위화하지 않습니다.',
             '다음 후보: 별도 실행에서 16/32회 학습, 동일 시간·FLOPs 예산 A 대조군, 미로 생성 계열별/경로 길이별 평가, 길이·벽 밀도 분포를 통제한 새로운 테스트, 명목·유효 파라미터 역할을 더 맞춘 비교. 실패 결과를 버리거나 이번 실행의 조건을 변경하지 않습니다.', '']
    (run/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    render(parser.parse_args().run)
