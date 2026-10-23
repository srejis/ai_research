"""Plots, predetermined qualitative categories, aggregate tables and Korean report."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'A': 'A: single-step', 'B': 'B: final loss (8)', 'C': 'C: all-step loss (8)'}


def render(run):
    run = Path(run)
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    data = np.load(run / 'dataset.npz')
    conditions = f'64-128-64 shared MLP | noise={config["noise_probability"]} | epochs={config["epochs"]} | seeds={config["seeds"]}'
    aggregate = []
    for metric, unit in [('pixel_error', 'Pixel error fraction'), ('bce', 'BCE (natural log)'),
                         ('change_mean', 'Mean absolute state change'), ('change_max', 'Max absolute state change')]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for mode in LABELS:
            runs = [s for s in summary if s['mode'] == mode]
            values = np.array([[r[metric] for r in s['metrics']] for s in runs])
            mean, std = values.mean(0), values.std(0, ddof=1 if len(runs) > 1 else 0)
            steps = np.arange(1, config['eval_steps'] + 1)
            line, = ax.plot(steps, mean, label=LABELS[mode])
            ax.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=.15)
            for row in values:
                ax.plot(steps, row, color=line.get_color(), alpha=.2, linewidth=.6)
        ax.set(xlabel='Inference iterations', ylabel=unit, title=conditions)
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(run / (metric + '.png'), dpi=160)
        plt.close(fig)
    for mode in LABELS:
        runs = [s for s in summary if s['mode'] == mode]
        for step in config['selected_steps']:
            row = {'mode': mode, 'step': step, 'n_seeds': len(runs)}
            for metric in ('bce', 'pixel_error', 'pixel_accuracy', 'exact_recovery', 'improved', 'worsened', 'change_mean'):
                values = [s['metrics'][step - 1][metric] for s in runs]
                row[metric + '_mean'] = float(np.mean(values))
                row[metric + '_std'] = float(np.std(values, ddof=1 if len(values) > 1 else 0))
            aggregate.append(row)
    (run / 'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    with (run / 'comparison.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    # Select first matching example in each category for the first seed, step1 -> step32.
    # Union of indices is used for ALL modes, including absent category annotations.
    indices, selection = set(), {}
    seed = config['seeds'][0]
    outputs = {m: np.load(run / f'{m}_seed{seed}_outputs.npz')['states'] for m in LABELS}
    for mode, states in outputs.items():
        errors = (states >= .5) != data['test_y'][None].astype(bool)
        delta = errors[-1].mean(1) - errors[0].mean(1)
        selection[mode] = {}
        for category, mask in [('improved', delta < 0), ('unchanged', delta == 0), ('worsened', delta > 0)]:
            matches = np.flatnonzero(mask)
            index = int(matches[0]) if len(matches) else None
            selection[mode][category] = index
            if index is not None:
                indices.add(index)
    (run / 'example_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    for mode, states in outputs.items():
        fig, axes = plt.subplots(len(indices), 8, figsize=(13, max(3, len(indices) * 1.5)), squeeze=False)
        for row, index in enumerate(sorted(indices)):
            images = [data['test_y'][index], data['test_x'][index]] + [states[s - 1, index] for s in config['selected_steps']]
            titles = ['Target', 'Noisy'] + [f'k={s}' for s in config['selected_steps']]
            for col, (image, title) in enumerate(zip(images, titles)):
                axes[row, col].imshow(image.reshape(8, 8), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                axes[row, col].set_title(f'#{index} {title}', fontsize=8)
                axes[row, col].axis('off')
        fig.suptitle(f'{LABELS[mode]}, seed={seed}: same indices across models; probability images')
        fig.tight_layout()
        fig.savefig(run / f'examples_{mode}.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode in LABELS:
        runs = [s for s in summary if s['mode'] == mode]
        for axis, key in zip(axes, ['model_only_ms_mean', 'end_to_end_ms_mean']):
            values = np.array([[t[key] for t in s['latency']] for s in runs])
            axis.errorbar(config['selected_steps'], values.mean(0), yerr=values.std(0), label=LABELS[mode])
            axis.set(xlabel='Inference iterations', ylabel='Latency (ms / batch)', title=key)
            axis.legend(fontsize=8)
    fig.suptitle(f'Synchronized warmup timings; batch={config["batch_size"]}; {config["device"]}')
    fig.tight_layout()
    fig.savefig(run / 'latency.png', dpi=160)
    plt.close(fig)
    report = ['# H002 / SH001 / EXP001 실제 실행 보고', '', f'실행: `{run.name}`. 조건: `{conditions}`.', '',
              '## 동일한 가중치를 반복 학습하고 반복 추론했을 때, 실제 정답 품질이 개선되었는가?', '',
              '아래 판단은 테스트 픽셀 오류율 기준이며 BCE도 함께 비교해야 합니다. ±는 seed 간 표본 표준편차입니다.', '',
              '| 방식 | 1회 오류율 | 8회 오류율 | 32회 오류율 | 1→8회 판단 |',
              '|---|---:|---:|---:|---|']
    for mode in LABELS:
        rows = {r['step']: r for r in aggregate if r['mode'] == mode}
        vals = [f'{rows[k]["pixel_error_mean"]:.6f} ± {rows[k]["pixel_error_std"]:.6f}' for k in (1, 8, 32)]
        delta = rows[8]['pixel_error_mean'] - rows[1]['pixel_error_mean']
        verdict = '평균 개선 관측' if delta < -1e-6 else '평균 악화 관측' if delta > 1e-6 else '정체 관측'
        report.append(f'| {mode} | ' + ' | '.join(vals) + f' | {verdict} |')
    report += ['', '## Seed별 실측 및 계산 비용', '',
               '| 모델 | 1회 오류율 | 8회 오류율 | 32회 오류율 | 학습 초 | optimizer steps | 학습 mapping calls | peak GPU MiB |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        memory = s['peak_gpu_allocated_bytes']
        report.append(f'| {s["name"]} | {s["metrics"][0]["pixel_error"]:.6f} | {s["metrics"][7]["pixel_error"]:.6f} | {s["metrics"][31]["pixel_error"]:.6f} | {s["train_seconds"]:.2f} | {s["optimizer_steps"]} | {s["training_mapping_calls"]} | {memory / 2**20 if memory else 0:.2f} |')
    report += ['', '## 안정성 관측', '']
    for s in summary:
        metrics = s['metrics']
        history = json.loads((run / (s['name'] + '_history.json')).read_text(encoding='utf-8'))
        tail = [h['loss'] for h in history[-10:]]
        report.append(f'- {s["name"]}: 최대 gradient norm={s["gradient_norm_max"]:.6g}, 마지막 {len(tail)}epoch loss 범위={min(tail):.6f}~{max(tail):.6f}, 32회 평균 상태 변화={metrics[-1]["change_mean"]:.6g}, period-2 비율={max(r["period2_fraction"] for r in metrics):.6f}, nonfinite={sum(r["nonfinite"] for r in metrics)}.')
    report += ['', '수치적 발산은 비유한 값/범위 이탈로 확인하며, 오차 악화와 구분합니다. Sigmoid로 유한 상태는 [0,1]에 제한됩니다. period-2 판정은 2단계 전 차이<1e-5 및 직전 차이>1e-4인 샘플 비율입니다. 다른 주기나 미세 진동의 부재를 증명하지 않습니다.', '',
               '## 데이터 난이도와 모호성', '',
               '원본은 1~4개 직선/직사각형을 결합한 뒤 좌우 대칭으로 만듭니다. 서로 다른 원본 6,144개를 생성 순서대로 분리하고 노이즈는 split별 독립 seed를 사용합니다. 원본 생성 시도에서 중복을 제거하므로 이 생성기의 고유 패턴 분포를 평가합니다.',
               '좌우 대칭은 복원 단서를 제공하지만, 서로 다른 두 관측 픽셀만으로 어느 쪽이 손상되었는지 확정할 수 없습니다. 양쪽이 함께 반전되는 경우도 있습니다. 15% 반전에서는 모든 관측이 여러 원본에서 발생할 수 있어 정답의 완전 식별은 보장되지 않습니다. 데이터 감사의 입력 오류율·0 예측 오류율·대칭쌍 불일치율·관측 충돌 수를 참고하십시오. 무작위 손상 입력 유지(약 15% 오류)가 복원 난이도의 직접 기준입니다.', '',
               '## 해석과 한계', '',
               '1→8회 개선과 A/B/C 간 성능 차이는 별개입니다. 평균 개선이 있어도 단계별 단조 개선이나 모든 샘플의 개선을 뜻하지 않습니다. 1~32회 전체 지표의 improved/worsened/unchanged는 바로 직전 단계와 비교하며, 1회는 입력과 비교합니다.',
               'A 대비 B/C는 optimizer step은 같아도 학습 mapping 호출이 8배입니다. 학습 시간은 배치 순서 생성·검증·손실/gradient 기록을 포함하고 최종 테스트·체크포인트 저장·시각화는 제외합니다. peak 메모리는 학습 구간 allocated 기준이며 전용 GPU 점유량이 아닙니다. 지연 시간은 warm-up 후 동기화된 wall time이고 모델 전용과 H2D+모델+D2H를 분리합니다. 작은 모델이므로 Python launch overhead와 시스템 부하의 영향을 받습니다.',
               '동일 계산 예산의 대조 모델을 실행하지 않았으므로 연산 효율의 우월성은 미확정입니다. 학습 부족·정보 손실·손실 목표 부적합성은 이 비교만으로 인과적으로 확정할 수 없습니다. 상태 변화가 작으면 수렴 경향을 관측할 수 있지만 올바른 정답으로의 수렴을 뜻하지 않습니다. 로봇의 지속적 사고 능력은 검증하지 않았습니다.', '',
               '## 검증과 산출물', '',
               '`verification.json`: 16,576개 파라미터, 공유 모듈 8회 호출, 모든 중간 상태의 gradient, 데이터 재생성·중복 검사, 전체 체크포인트 복원 출력 비교 및 H001·공용 data SHA-256 보존 검사. `summary.json`, `*_history.json`, `*_metrics.json`, `*_latency.json`, `*_outputs.npz`, `dataset.npz`, `comparison.csv`, `aggregate.json`, 그래프 8개 및 source/ 스냅샷을 보존합니다. checkpoints/<run_id>/에는 optimizer 상태를 포함한 모델을 보존합니다.', '',
               '## 다음 실험 후보', '',
               '동일 wall-time/연산 예산의 A 대조군, 노이즈 확률별 난이도, 학습 반복 길이 2/4/16, 고정점과 샘플별 실패 유형 분석을 별도 실행으로 비교할 수 있습니다. 안정화 구조 변경은 이번 원래 조건과 구분해야 합니다.', '']
    (run / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    render(parser.parse_args().run)
