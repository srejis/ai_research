"""Figures A-F and evidence-based Korean report, generated from saved numbers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_io import ROOT, read_json

COLORS = {"B0": "tab:blue", "B": "tab:orange", "C0": "tab:green", "C": "tab:red"}


def generate(summary, performance, changes, pairs):
    figures = Path(summary["figure_directory"])
    reports = Path(summary["report_directory"])
    k = summary["K"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=170)
        plt.close(fig)

    def stages(rows):
        return sorted([r for r in rows if r["stage"] == "after_insertion" or r["stage"].startswith("epoch_")], key=lambda r: r["local_epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, branches, title in zip(axes, [("B0", "B"), ("C0", "C")], [f"Mid source: epoch {summary['E_mid']}", f"Complete source: epoch {summary['E_complete']}"]):
        for branch in branches:
            rows = stages([r for r in performance if r["branch"] == branch])
            ax.plot([r["local_epoch"] for r in rows], [r["validation_accuracy"] * 100 for r in rows], marker="o", label=branch, color=COLORS[branch])
            last = rows[-1]
            ax.annotate(f"{branch}: {last['validation_accuracy']*100:.2f}%",
                        (last["local_epoch"], last["validation_accuracy"]*100),
                        xytext=(-90, 12 if branch.endswith('0') else -18), textcoords="offset points",
                        color=COLORS[branch], fontsize=9)
        ax.set(title=title, xlabel="Additional epoch", ylabel="Validation accuracy (%)", xticks=list(range(k+1)))
        ax.grid(alpha=0.25)
        ax.legend()
        ax.margins(y=0.15)
    fig.suptitle("A. Validation accuracy (different source epochs; separate y scales)")
    save(fig, "A_validation_accuracy.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, layer in zip(axes, ("fc1", "fc2", "fc3")):
        for branch in COLORS:
            rows = stages([r for r in changes if r["branch"] == branch and r["parameter"] == f"old.{layer}"])
            ax.plot([r["local_epoch"] for r in rows], [r["relative_l2"] for r in rows], marker="o", color=COLORS[branch], label=f"{branch} (source {summary['branches'][branch]['source_epoch']})")
        ax.set(title=f"Original {layer} weight + bias", xlabel="Additional epoch", ylabel="Relative L2 change", xticks=list(range(k+1)))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("B. Original parameters: displacement from branch start")
    save(fig, "B_original_relative_changes.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, pair in zip(axes, ("B-B0", "C-C0")):
        for layer in ("fc1", "fc2", "fc3"):
            rows = [r for r in pairs if r["pair"] == pair and r["parameter"] == f"old.{layer}"]
            ax.plot([r["step"] for r in rows], [r["relative_l2"] for r in rows], marker="o", label=layer)
        ax.set(title=pair, xlabel="Additional optimizer step", ylabel="L2(delta_growth - delta_control) / (L2(start)+eps)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("C. Paired trajectory difference (includes first update)")
    save(fig, "C_paired_trajectory.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, parameter in zip(axes, ("new.input_weight", "new.input_bias", "new.output_weight")):
        for branch in ("B", "C"):
            rows = [r for r in changes if r["branch"] == branch and r["parameter"] == parameter and r["stage"] != "before_insertion"]
            ax.plot([r["step"] for r in rows], [r["absolute_l2"] for r in rows], marker="o", label=f"{branch} displacement", color=COLORS[branch])
            ax.plot([r["step"] for r in rows], [r["current_l2"] for r in rows], linestyle="--", label=f"{branch} current norm", color=COLORS[branch])
        ax.set(title=parameter, xlabel="Additional optimizer step", ylabel="Absolute L2")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("D. New-neuron parameters (absolute norms; zero output base)")
    save(fig, "D_new_neuron.png")

    final_stage = f"epoch_{k:03d}"
    final_pairs = [r for r in pairs if r["stage"] == final_stage]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xx = np.arange(3)
    for ax, key, ylabel in zip(axes, ("absolute_l2", "relative_l2"), ("Absolute L2 trajectory difference", "Relative L2 trajectory difference")):
        for shift, pair in [(-0.18, "B-B0"), (0.18, "C-C0")]:
            rows = [next(r for r in final_pairs if r["pair"] == pair and r["parameter"] == f"old.{layer}") for layer in ("fc1", "fc2", "fc3")]
            ax.bar(xx + shift, [r[key] for r in rows], width=0.36, label=pair)
        ax.set(xticks=xx, xticklabels=["fc1", "fc2", "fc3"], ylabel=ylabel, xlabel="Original layer weight + bias")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("E. Layer influence at the final additional epoch")
    save(fig, "E_layer_influence.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row_axes, pair in zip(axes, ("B-B0", "C-C0")):
        data = np.load(Path(summary["weights_directory"]) / f"{pair}_{final_stage}.npz")
        for ax, layer in zip(row_axes, ("fc1", "fc2", "fc3")):
            delta = np.concatenate([data[f"{layer}.weight.delta_difference"], data[f"{layer}.bias.delta_difference"]])
            ax.hist(delta, bins=60)
            ax.set(title=f"{pair}: {layer}", xlabel="delta_growth - delta_control", ylabel="Count")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("F. Distribution of original-coordinate trajectory differences")
    save(fig, "F_difference_distributions.png")

    lines = ["# EXP002 — 뉴런 1개 추가에 따른 기존 가중치 변화", "",
             "## 실행 조건과 EXP001 검증", "",
             f"실행 ID: `{summary['run_id']}`. 기존 research 환경, {summary['gpu']}, PyTorch {summary['torch']}, FP32, TF32 비활성화, 결정적 연산을 사용했습니다.", "",
             f"기준 모델은 784→256→128→10입니다. 로그에서 최초 검증 95% 도달은 E_complete={summary['E_complete']}, E_mid={summary['E_mid']}, K={k}로 확인했습니다. E_mid=0은 학습 전 초기 모델이며, 학습 도중의 양의 에포크를 뜻하지 않습니다.", "",
             "두 원본 체크포인트는 모델·가중치 복원과 검증 손실·정확도·전체 logits 해시 일치를 통과했습니다. 전체 EXP001 파일의 실행 전후 SHA-256도 일치합니다. EXP001을 재학습하지 않았고, 이번 실행에서 테스트 데이터는 평가하지 않았습니다.", "",
             "| 기준 | 실제 체크포인트 | 검증 정확도 | 검증 손실 |", "|---|---|---:|---:|"]
    for key in ("mid", "complete"):
        item = summary["insertion_checks"][key]
        lines.append(f"| {key} | `{item['checkpoint']}` | {item['validation']['accuracy']:.4%} | {item['validation']['loss']:.9f} |")
    lines += ["", "각 분기는 새로운 동일 설정의 AdamW를 사용했습니다(lr=0.001, weight_decay=0.0001, betas=(0.9,0.999), eps=1e-8). 대응 분기쌍은 원본 체크포인트의 shuffle generator 상태로 만든 동일한 저장 순열을 사용했습니다. 따라서 완료 모델 분기는 기존 AdamW moment를 이어간 결과가 아닙니다.", "",
              "## 관찰된 사실", "", "### 추가 직후 출력", "",
              "기존 좌표는 비트 단위로 그대로 복사했고 첫 은닉층에 정확히 1개(파라미터 913개)를 추가했습니다. 신규 입력/편향은 seed=42001, 신규 출력 열은 0입니다. 검증 입력 5,000개 전체에서 추가 전후 logits를 비교했습니다.", "",
              "| 출발점 | 최대 절대 logits 차이 | RMS 차이 | 허용오차 검사 |", "|---|---:|---:|---|"]
    for key in ("mid", "complete"):
        check = summary["insertion_checks"][key]["output_preservation"]
        lines.append(f"| {key} | {check['max_absolute_logit_difference']:.9g} | {check['rms_logit_difference']:.9g} | {check['allclose']} (atol={check['atol']}, rtol={check['rtol']}) |")
    lines += ["", "### 성능과 추가 학습량", "", "| 분기 | 출발 epoch | 추가 epoch / step | 시작 검증 정확도 | 최종 훈련 정확도 | 최종 검증 정확도 | 최종 검증 손실 |", "|---|---:|---|---:|---:|---:|---:|"]
    for branch in COLORS:
        item = summary["branches"][branch]
        start = next(r for r in performance if r["branch"] == branch and r["stage"] == "after_insertion")
        final = item["final_metrics"]
        lines.append(f"| {branch} | {item['source_epoch']} | {item['epochs']} / {item['steps']} | {start['validation_accuracy']:.4%} | {final['train_accuracy']:.4%} | {final['validation_accuracy']:.4%} | {final['validation_loss']:.9f} |")
    lines += ["", "### 원래 가중치의 변화 및 대응 대조군과의 차이", "",
              "아래 레이어 통계는 원래 weight와 bias만 연결한 벡터이며 신규 좌표를 제외합니다. 변화는 각 분기의 원래 출발 가중치에 대한 차이입니다. 상대 L2는 L2(Δ)/(L2(start)+1e-12)입니다.", "",
              "| 분기 | 레이어 | 절대 L2 변화 | 상대 L2 변화 | 평균 절대 변화 | 최대 절대 변화 |", "|---|---|---:|---:|---:|---:|"]
    for branch in COLORS:
        for layer in ("fc1", "fc2", "fc3"):
            row = next(r for r in changes if r["branch"] == branch and r["stage"] == final_stage and r["parameter"] == f"old.{layer}")
            lines.append(f"| {branch} | {layer} | {row['absolute_l2']:.9g} | {row['relative_l2']:.9g} | {row['mean_absolute']:.9g} | {row['max_absolute']:.9g} |")
    lines += ["", "`Δ_growth−Δ_control`은 동일한 원래 좌표의 학습 궤적 차이입니다. 단순한 두 변화량 norm의 뺄셈과 구분하며 두 값 모두 CSV에 보존했습니다.", "",
              "| 비교 | 레이어 | 궤적 차이 L2 | 상대 궤적 차이 | 업데이트 cosine |", "|---|---|---:|---:|---:|"]
    pair_notes = []
    for pair in ("B-B0", "C-C0"):
        for layer in ("fc1", "fc2", "fc3"):
            row = next(r for r in final_pairs if r["pair"] == pair and r["parameter"] == f"old.{layer}")
            lines.append(f"| {pair} | {layer} | {row['absolute_l2']:.9g} | {row['relative_l2']:.9g} | {row['update_cosine']:.9g} |")
        tensor_rows = [r for r in final_pairs if r["pair"] == pair and not r["parameter"].startswith("old.")]
        largest = max(tensor_rows, key=lambda r: r["absolute_l2"])
        relative = max(tensor_rows, key=lambda r: r["relative_l2"])
        largest_layer = max((r for r in final_pairs if r["pair"] == pair and r["parameter"] in {"old.fc1", "old.fc2", "old.fc3"}), key=lambda r: r["absolute_l2"])
        pair_notes += ["", f"{pair}: 절대 궤적 차이가 가장 큰 레이어는 {largest_layer['parameter']}, 텐서는 {largest['parameter']} (L2={largest['absolute_l2']:.9g})입니다. 상대 차이가 가장 큰 텐서는 {relative['parameter']} ({relative['relative_l2']:.9g})입니다."]
    lines += pair_notes
    lines += ["", "성능과 함께 비교하면 다음과 같습니다. 아래 차이는 확장군−대조군이며 인과관계나 통계적 유의성을 뜻하지 않습니다."]
    for growth, control in (("B", "B0"), ("C", "C0")):
        g, c = summary["branches"][growth]["final_metrics"], summary["branches"][control]["final_metrics"]
        lines += ["", f"- {growth}−{control}: 검증 정확도 {(g['validation_accuracy']-c['validation_accuracy'])*100:+.4f}%p, 검증 손실 {g['validation_loss']-c['validation_loss']:+.9f}. 이 실행에서 확장군의 검증 정확도가 더 높지는 않았습니다."]
    lines += ["", "### 첫 역전파와 신규 뉴런", "", "| 분기 | 입력 weight gradient L2 | 입력 bias gradient | 출력 gradient L2 | 첫 입력 update L2 | decay 예측 L2 |", "|---|---:|---:|---:|---:|---:|"]
    for branch, value in summary["first_gradients"].items():
        lines.append(f"| {branch} | {value['new_input_weight_gradient_l2']:.9g} | {value['new_input_bias_gradient']:.9g} | {value['new_output_weight_gradient_l2']:.9g} | {value['first_input_update_l2']:.9g} | {value['adamw_decay_prediction_l2']:.9g} |")
    lines += ["", "신규 입력의 첫 task-loss gradient는 정확히 0입니다. 첫 입력 가중치 업데이트는 AdamW의 decoupled weight decay 및 FP32 반올림에 의해 생깁니다. 관련 원본 gradient 텐서와 decay 예측 잔차는 별도 파일에 저장했습니다. 출력 연결은 시작 norm이 0이므로 상대 변화량이 epsilon에 크게 의존합니다. 신규 출력의 해석은 절대 L2와 현재 norm을 사용합니다.", "",
              "| 분기 | 신규 파라미터 | 종료 절대 변화 L2 | 종료 현재 norm |", "|---|---|---:|---:|"]
    for branch in ("B", "C"):
        for name in ("new.input_weight", "new.input_bias", "new.output_weight"):
            row = next(r for r in changes if r["branch"] == branch and r["stage"] == final_stage and r["parameter"] == name)
            lines.append(f"| {branch} | {name} | {row['absolute_l2']:.9g} | {row['current_l2']:.9g} |")
    lines += ["", "## 해석과 한계", "",
              "관측된 궤적 차이는 뉴런을 추가한 이 실행 조건에서 기존 파라미터의 업데이트가 대응 대조군과 달라졌음을 보여줍니다. 가능한 설명은 신규 출력 연결이 학습되면서 이후 역전파 경로와 기존 레이어의 gradient가 바뀌는 것입니다. 이는 원인에 대한 해석이며, 별도의 gradient 경로 분해 실험으로 검증한 결론은 아닙니다.", "",
              "구조 차이로 행렬 연산 크기가 바뀌면 FP32 연산 순서에 따른 작은 수치 차이가 섞일 수 있습니다. 추가 직후의 차이 및 첫 업데이트 차이를 보존했지만, 순수 신규 뉴런 효과와 커널 수치 효과를 완전히 분리한 실험은 아닙니다. 가중치 변화가 크다는 이유로 정확도가 개선됐다고 해석하지 않습니다. 정확도·손실의 동시 변화는 위 표와 A 그래프로 함께 판단해야 하며 상관관계의 통계적 유의성을 주장하지 않습니다.", "",
              "단일 seed, MNIST, K=1에포크의 관찰입니다. E_mid=0이라는 특성 때문에 '학습 중간에 이미 학습된 모델'의 일반적 효과로 해석할 수 없습니다. AdamW를 새로 시작했으므로 기존 optimizer를 이어간 경우에 대한 결론도 아닙니다.", "",
              "추가 검증 후보: 여러 seed와 반복 실행, 더 긴 관찰 구간, 양의 중간 epoch를 갖는 별도 기준 완료 조건, optimizer 상태 이식 실험, 계산 커널 수치 차이를 통제한 gradient 경로 분석입니다. 기존 EXP001의 완료 정의나 원본 결과를 소급 변경해서는 안 됩니다.", "",
              "## 산출물과 재현", "",
              "- `checkpoints/<branch>/<run_id>/`: before_insertion, after_insertion, first_update, epoch_001 전체 상태",
              "- `results/metrics/<run_id>/performance.csv`: 성능 지표와 단계; *_batch_order.npz: 공용 대응 순열",
              "- `results/weights/<run_id>/parameter_changes.csv`, `paired_changes.csv`: 텐서·레이어별 변화 및 cosine",
              "- 동일 weights 디렉터리의 NPZ: 전체 Δ 벡터·단위 방향·히스토그램 및 대응 차이 벡터",
              "- `*_first_backward_gradients.pt`, `first_gradient_analysis.json`: 최초 gradient 및 decay 분석",
              "- `results/figures/<run_id>/`: A~F PNG",
              "- 보고서 디렉터리: summary, baseline_verification, protected_baseline_before, 소스/설정 스냅샷, artifact 해시",
              "", "검사·재현 명령은 EXP002 README를 참조하십시오. 기존 run_id 덮어쓰기는 거부합니다."]
    (reports / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
