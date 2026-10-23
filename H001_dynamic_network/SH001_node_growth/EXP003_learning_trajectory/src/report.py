"""Korean research report: explicit observations, interpretation, and limits."""
import shutil
from pathlib import Path
import numpy as np

from core import json_read


def f(value,digits=5):
    return "미관측/결측" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def render(output,run_root,seeds,summaries,pairs,layers,aggregate,thresholds,new):
    conditions=("P25","P50","P75","P100")
    def summary(condition,metric):
        r=next(r for r in aggregate if r["condition"]==condition and r["metric"]==metric)
        return f"{f(r['mean'])} ± {f(r['sample_std'])}"
    lines=["# EXP003 — 뉴런 1개 추가에 따른 학습 방향·속도·경로 변화", "",
        "## 실험과 검증 개요", "",
        "EXP001·EXP002의 코드, 설정, 체크포인트와 결과는 수정하지 않았습니다. 각 seed 실행 전후와 독립 검증에서 전체 파일 SHA-256이 일치했습니다. 기존 Anaconda research 환경을 사용했고 새 환경 생성이나 추가 설치는 없었습니다. 실제 패키지 버전과 소스 해시를 seed별로 기록했습니다.", "",
        "새 기준 모델은 784→32→16→10입니다. MNIST 55,000/5,000 고정 분할과 정규화는 EXP001과 같습니다. AdamW(lr=0.001, weight_decay=0.0001, betas=(0.9,0.999), eps=1e-8), batch=128, ReLU, CrossEntropyLoss, FP32, TF32 비활성화, 결정적 연산을 사용했습니다. 모델 seed는 42/123/2026이고 신규 뉴런 초기화 seed는 42000+모델 seed입니다. 테스트 데이터는 사용하지 않았습니다.", "",
        "| seed | 최초 25-step 관측 ≥95% | 직전 구간 재생의 최초 도달 | 전 구간 점검 S_complete | P25 / P50 / P75 / P100 |", "|---|---:|---:|---:|---|"]
    for seed in seeds:
        completion=json_read(run_root/f"seed_{seed}/baseline/completion.json")
        lines.append(f"| {seed} | {completion['first_coarse_hit']} | {completion['first_hit_in_refined_bracket']} | {summaries[seed]['S_complete']} | {' / '.join(str(v) for v in summaries[seed]['branch_steps'].values())} |")
    lines += ["", "기준 학습은 seed별 하나의 경로를 생성했습니다. 25스텝 간격 탐색 후 직전 체크포인트에서 optimizer·RNG·배치 위치를 복원해 매 스텝 재평가했습니다. 그보다 이전에 잠시 95%를 넘었을 가능성도 검사하기 위해, 저장된 전체 FP32 가중치 경로를 forward 연산만으로 재평가했습니다. 위 S_complete는 모든 정수 step에 대해 확인한 최초 도달입니다. 목표 도달은 수렴의 정의가 아닙니다.", "",
        "분기점의 전체 상태는 직전 주기 체크포인트의 상태를 재생해 복원했고, 모든 재생 step에서 원래 가중치 경로와 비트 단위 일치를 확인했습니다. 이 복원용 재생과 검증 비용은 기본 학습 시간 비교에 포함하지 않았습니다. 탐색의 초과 구간과 복원 기록은 삭제하지 않았습니다.", "",
        "모든 분기점은 양수이며 중복되지 않았습니다. 각 분기의 추가 관측 구간은 해당 seed의 S_complete 스텝입니다. 대응 분기는 매 스텝 같은 샘플을 같은 순서로 처리했습니다. 새 뉴런 초기화는 별도 RNG 범위에서 수행해 셔플·공통 학습 난수 상태에 영향을 주지 않습니다.", "",
        "### 구조와 optimizer 상태 보존", "",
        "기존 fc1/fc2/fc3 파라미터를 그대로 유지하고, 784→1의 새 입력 및 1→16의 새 출력 경로를 fc2 pre-activation에 더했습니다. 이는 수학적으로 첫 은닉층 32→33 확장과 같으며, 뉴런은 정확히 1개·파라미터는 801개 추가됩니다. 기존 좌표는 복사 후 정확히 같고, 기존 AdamW의 exp_avg·exp_avg_sq·step도 같음을 검사했습니다. 두 optimizer의 CPU step 텐서가 공유되지 않도록 상태를 deep copy했습니다.", "",
        "새 파라미터는 별도 optimizer group에 두고 같은 AdamW 설정과 weight decay를 적용했습니다. 신규 상태는 첫 step에 0부터 시작하지만 기존 상태의 step은 원래 학습 step을 이어갑니다. 처음 입력 task gradient가 0이어도 AdamW decay는 적용됩니다.", "",
        "추가 직후 logits는 모든 검증 입력에서 비트 단위로 같았습니다. 통합된 33뉴런 행렬 구현과도 수치 허용오차 내에서 같음을 별도로 확인했습니다. 원래 레이어의 연산 모양을 유지하므로, 확장 직후 기존 행렬 커널의 크기 변경으로 생길 수 있는 차이를 줄였습니다.", "",
        "## Q1. 기존 가중치의 업데이트 방향은 실제로 달라지는가?", "",
        "**측정:** 원래 weight와 bias만 레이어별로 연결해, 실제 매 스텝 업데이트와 task-loss gradient의 cosine을 각각 계산했습니다. 새 뉴런은 이 비교에서 제외했습니다. 두 norm 중 하나라도 1e-12 이하면 cosine은 결측으로 처리했습니다. 사전 정의한 큰 방향 차이 기준은 cosine<0.99(약 8.1도 초과)이며 보편적 경계가 아닙니다.", ""]
    old=[r for r in layers if r["layer"]!="all"]
    detected=[r for r in old if r["first_update_cosine_below_099"] is not None]
    firsts=[r["first_update_cosine_below_099"] for r in detected]
    lines += [f"**관찰:** 최초 업데이트에서는 기존 task gradient와 실제 업데이트가 대조군과 정확히 같았습니다. 이후 {len(old)}개 seed·시점·레이어 조합 중 {len(detected)}개에서 업데이트 cosine<0.99를 관측했습니다. 관측된 최초 시점의 범위는 추가 {f(min(firsts) if firsts else None)}~{f(max(firsts) if firsts else None)}스텝입니다.", "",
        "| seed | 시점 | 레이어 | update cos<0.99 최초 | gradient cos<0.99 최초 | 평균 update cosine | 종료 누적 cosine | 종료 정규화 거리 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in old:
        lines.append(f"| {r['seed']} | {r['condition']} | {r['layer']} | {f(r['first_update_cosine_below_099'])} | {f(r['first_gradient_cosine_below_099'])} | {r['mean_update_cosine']:.5f} | {r['final_cumulative_cosine']:.5f} | {r['final_relative_path_distance']:.5g} |")
    lines += ["", "**해석:** 신규 출력 연결이 학습되면 손실의 미분 경로와 이후 기존 파라미터의 gradient가 달라질 수 있습니다. AdamW의 누적 moment 때문에 gradient 방향 차이와 실제 업데이트 방향 차이의 크기·발생 시점은 같지 않을 수 있습니다. 이를 동일한 지표로 취급하지 않았습니다.", "",
        "방향 차이의 일시성은 `direction_seed_metrics.csv`의 처음/마지막 사분위 구간 평균, cosine<0.99인 step 비율과 그림 1의 생값·25스텝 이동평균으로 확인합니다. 누적 방향은 그림 7, 경로 거리는 그림 4로 별도 제시했습니다. 최종 거리만으로 학습 방향을 추정하지 않습니다.", "",
        "## Q2. 뉴런 추가는 학습 속도를 변화시키는가?", "",
        "아래는 같은 추가 step 수에서의 최종 차이입니다. 정확도 차이의 단위는 %p이고, loss 차이는 Growth−Control입니다. ±는 3개 seed의 표본 표준편차(ddof=1)이며 신뢰구간이 아닙니다.", "",
        "| 시점 | 정확도 차이 평균±SD (%p) | loss 차이 평균±SD | 공통 시간 예산 loss 차이 | 공통 MAC proxy 예산 loss 차이 |", "|---|---:|---:|---:|---:|"]
    for condition in conditions:
        lines.append(f"| {condition} | {summary(condition,'accuracy_difference_pp')} | {summary(condition,'loss_difference')} | {summary(condition,'equal_time_loss_difference')} | {summary(condition,'equal_mac_loss_difference')} |")
    lines += ["", "| seed | 시점 | Control 정확도 | Growth 정확도 | G−C loss | Control 학습 초 | Growth 학습 초 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"| {r['seed']} | {r['condition']} | {r['control_accuracy']:.4%} | {r['growth_accuracy']:.4%} | {r['loss_difference']:+.7f} | {r['control_train_wall_seconds']:.3f} | {r['growth_train_wall_seconds']:.3f} |")
    lines += ["", "각 조건에서 최종 loss가 낮았던 반복 수는 다음과 같습니다. 이는 특정 종료 시점의 관찰이며 전체 과정의 일관된 우위를 뜻하지 않습니다.", ""]
    for condition in conditions:
        subset=[r for r in pairs if r["condition"]==condition]
        lines.append(f"- {condition}: 같은 step에서 {sum(r['loss_difference']<0 for r in subset)}/{len(seeds)}회, 공통 시간 예산에서 {sum(r['equal_time_loss_difference']<0 for r in subset)}/{len(seeds)}회 Growth의 loss가 더 낮았습니다.")
    lines += ["", "### 목표 정확도 도달 step", "",
        "검증은 대응 분기에서 매 스텝 수행했습니다. 90/95/96/97%에 이미 도달한 출발 상태는 추가 step=0이고, 관측 종료까지 도달하지 못한 경우 결측입니다. 미달성을 0이나 성공으로 바꾸지 않습니다. 아래 step 차이는 두 조건 모두 도달한 seed만의 평균±SD이며, 선택된 표본 수를 함께 표시합니다.", "",
        "| 시점 | 목표 | Control 도달 seed | Growth 도달 seed | 대응 표본 수 | Growth−Control step 평균±SD |", "|---|---:|---:|---:|---:|---:|"]
    for r in thresholds:
        lines.append(f"| {r['condition']} | {r['target']:.0%} | {r['control_reached_seeds']} | {r['growth_reached_seeds']} | {r['paired_achieved_seeds']} | {f(r['growth_minus_control_steps_mean'])} ± {f(r['sample_std'])} |")
    lines += ["", "**시간 측정의 한계:** CUDA synchronize를 사용한 forward/backward 및 optimizer 구간 wall time을 합산했습니다. Control/Growth의 실행 순서는 매 step 교대로 바꿨습니다. gradient 복사, 지표 계산, 검증, 파일 쓰기는 이 학습 구간 시간에서 제외했고, 전체 계측 실행 시간은 각 pair summary에 별도로 기록했습니다. GPU event 시간도 CSV에 보존했습니다. 작은 모델의 launch·동기화 비용과 계측 영향이 있으므로, 계측 없는 전용 성능 benchmark와 같은 결과라고 해석하지 않습니다.", "",
        "**연산량의 한계:** forward dense-MAC proxy는 처리 샘플 수×25,760(Control), ×26,560(Growth)입니다. Growth의 proxy는 약 3.11% 크며 bias·ReLU·backward·optimizer·실제 커널 비용을 포함한 전체 FLOPs가 아닙니다. 동일 예산 비교는 두 조건의 공통 최대 시간 또는 proxy 이하에서 마지막으로 관측된 step을 선택했습니다. 정확히 같은 시간이나 전체 FLOPs를 사용했다는 주장은 하지 않습니다.", "",
        "Train loss는 같은 batch를 업데이트하기 전에 측정한 task loss입니다. 검증 지표는 업데이트 후 전체 검증 데이터에서 측정했습니다. 따라서 서로 다른 batch의 train loss가 출렁이는 것을 학습 속도 자체로 해석하지 않습니다. 각 seed·조건별 loss 감소율과 처리 샘플 수, 시간·비용 비교는 `paired_seed_metrics.csv`에 있습니다.", "",
        "## Q3. 구조 변경의 효과는 추가 시점에 따라 달라지는가?", "",
        "P25/P50/P75/P100별 방향 차이 발생 시점, 누적 경로 거리, 성능 차이와 목표 도달 속도를 위 표와 과정 그래프로 비교합니다. 서로 다른 조건의 출발 상태 및 종료 절대 step은 다릅니다. 따라서 최종 정확도만으로 최적 추가 시점을 결정하지 않습니다. 같은 시점에서도 seed에 따라 개선 여부가 바뀌면 일관된 이득으로 일반화할 수 없습니다.", ""]
    best_loss=min((r for r in aggregate if r["metric"]=="loss_difference"),key=lambda r:r["mean"])
    best_accuracy=max((r for r in aggregate if r["metric"]=="accuracy_difference_pp"),key=lambda r:r["mean"])
    lines += [f"**관찰:** 같은 추가 step 수의 최종 평균 loss 차이가 가장 낮은 조건은 {best_loss['condition']}({f(best_loss['mean'])}), 평균 정확도 차이가 가장 큰 조건은 {best_accuracy['condition']}({f(best_accuracy['mean'])}%p)입니다. 이는 관측한 네 조건의 기술적 순위이며 최적 시점의 검증은 아닙니다.", "",
        "## Q4. 새 뉴런은 실제 학습에 참여했는가?", "",
        "| seed | 시점 | 입력 gradient 첫 비영 step | 출력 gradient 첫 비영 step | 종료 입력 weight norm | 종료 출력 weight norm | 마지막 batch 활성 비율 | 마지막 batch 상대 기여 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for condition in conditions:
            info=summaries[seed]["pairs"][condition]
            end=new[seed,condition][-1]
            lines.append(f"| {seed} | {condition} | {f(info['first_nonzero_new_gradient']['input'])} | {f(info['first_nonzero_new_gradient']['output'])} | {end['input_weight_norm']:.6g} | {end['output_weight_norm']:.6g} | {end['relu_active_fraction']:.4f} | {end['contribution_relative_norm']:.6g} |")
    lines += ["", "첫 입력 gradient는 0임을 확인했습니다. 그동안 입력 weight는 AdamW decay로 변할 수 있습니다. 출력 연결이 학습된 뒤 입력 gradient도 관측되면, 초기의 0 gradient를 학습 실패로 판단하지 않습니다. 활성·기여 값은 업데이트 전 동일 훈련 batch의 값이고, weight norm은 업데이트 후 값입니다. 상대 기여는 ||new contribution||/(||old fc2 preactivation||+1e-12)이며 인과적 중요도 점수가 아닙니다.", "",
        "## Q5. 방향 변화는 성능 변화와 관련 있는가?", "",
        "각 대응 경로에서 1−실제 업데이트 cosine(all)과 Growth−Control 검증 loss 차이의 기술적 Pearson 상관을 계산했습니다. 아래 양의 상관은 방향 차이가 큰 step에서 Growth의 상대 loss가 더 높았다는 기술적 관계이고, 음의 상관은 그 반대입니다. 두 시계열은 자기상관이 있고 시간에 따른 변화라는 공통 요인이 있으므로 유의성·인과관계를 주장하지 않습니다.", "",
        "| seed | 시점 | 전체 기존 파라미터 평균 update cosine | 방향 차이와 loss 차이의 Pearson r | 종료 G−C loss | 종료 정확도 차이(%p) |", "|---|---|---:|---:|---:|---:|"]
    for r in pairs:
        direction=next(x for x in layers if x["seed"]==r["seed"] and x["condition"]==r["condition"] and x["layer"]=="all")
        lines.append(f"| {r['seed']} | {r['condition']} | {direction['mean_update_cosine']:.5f} | {f(r['direction_vs_loss_gap_pearson_descriptive'])} | {r['loss_difference']:+.7f} | {r['accuracy_difference_pp']:+.4f} |")
    lines += ["",
        "큰 방향 차이·거리만으로 좋은 학습이라 판단하지 않습니다. 같은 step의 성능 차이와 같은 시간·연산 proxy 예산의 성능 차이를 구분합니다. 위 seed별 결과와 조건별 개선 반복 수를 근거로 판단해야 하며, 반대 부호의 반복을 평균 하나로 감추지 않았습니다.", "",
        "## 확인하지 못한 부분과 후속 과제", "",
        "3개 seed는 반복 관찰이며 일반화를 위한 충분한 표본 수라고 볼 수 없습니다. 신규 경로를 분리한 구현은 통합 Linear 구현과 커널 실행 비용이 다를 수 있습니다. 신규 뉴런의 초기화 seed를 독립적으로 여러 번 바꾸는 반복, 더 긴 관측 구간, 뉴런 추가 위치 변경, 계측을 제거한 별도 시간 benchmark가 후속 과제입니다.", "",
        "## 원본 수치와 재현", "",
        "각 seed의 baseline/steps.csv, validation_coarse.csv, completion_refinement.csv, validation_exact.csv와 branches/*/{performance,directions,new_neuron}.csv를 보존했습니다. initial.npy와 25스텝 단위 XOR 청크로 모든 FP32 파라미터를 비트 손실 없이 복원하므로 실제 업데이트를 float64로 재계산할 수 있습니다. 전체 AdamW·RNG·배치 순서와 위치는 주기·분기·최초·종료 체크포인트에서 복원할 수 있습니다.", "",
        "독립 검증은 원본 해시, 이전 두 실험 보존, 전 구간 최초 도달, optimizer 상태 매핑, 동일 배치 순서, CPU 체크포인트 로드, 검증 출력 재현, 매 스텝 업데이트 방향·거리의 독립 재계산, 첫 task gradient·업데이트 일치 및 중첩 Control 경로의 비트 일치를 포함했습니다. raw gradient 벡터는 첫 step에 보존하고, 이후에는 매 step 방향과 norm을 기록했습니다.", "",
        "서로 다른 seed의 파라미터 좌표는 직접 비교하지 않았습니다. 그래프의 normalized additional step은 seed별 추가 step/S_complete이며, 평균·표준편차 곡선은 이 축에서 선형 보간한 기술 통계입니다. raw 그래프와 개별 seed 표, 원본 CSV를 함께 보존했습니다. 실행 명령과 파일 구조는 EXP003 README를 참조하십시오."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    shutil.copy2(Path(__file__),output/"report_source.py")
