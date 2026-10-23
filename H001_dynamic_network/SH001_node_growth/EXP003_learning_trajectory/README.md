# EXP003 — Learning Trajectory

축소 MLP에 뉴런 1개를 추가했을 때 기존 가중치의 매 스텝 업데이트 방향, 학습 속도와 경로가 어떻게 달라지는지 관찰합니다. EXP001·EXP002는 수정하지 않고 원본 데이터 분할·전처리만 참조합니다.

## 실험 설정

기준 모델은 `784→32→16→10`, 확장 모델은 수학적으로 `784→33→16→10`입니다. 기존 Linear 파라미터를 유지하는 별도 신규 뉴런 경로로 구현해 AdamW의 기존 moment와 step을 그대로 보존합니다. 새 파라미터는 같은 설정의 별도 optimizer group으로 관리하며 상태는 새로 시작합니다. 모든 파라미터에 동일한 AdamW decay를 적용합니다.

MNIST의 기존 55,000/5,000 고정 분할 및 정규화 `(pixel/255−0.1307)/0.3081`을 사용합니다. AdamW(lr=0.001, weight_decay=0.0001, betas=(0.9,0.999), eps=1e-8), batch=128, ReLU, CrossEntropyLoss, FP32, TF32 off, deterministic이며 모델 seed는 42/123/2026입니다. 별도 신규 뉴런 seed는 42000+모델 seed입니다. 설정은 [configs/experiment.json](configs/experiment.json)에 있습니다.

검증 95% 최초 도달 step을 `S_complete`로 정의합니다. 25스텝 간격으로 탐색하고 직전 체크포인트에서 optimizer·RNG·배치 위치를 복원해 구간을 재생합니다. 이전 구간의 일시적 도달도 놓치지 않도록 lossless 가중치 경로를 학습 없이 전수 재평가합니다. 분기점은 floor(S_complete×0.25/0.5/0.75/1.0)이며 모두 양수·비중복이어야 합니다. 각 Control/Growth 분기쌍은 같은 배치로 S_complete만큼 추가 학습하고 매 스텝 검증합니다. 테스트 데이터는 사용하지 않습니다.

## 실행 및 재현

기존 `D:\anaconda3\envs\research` 환경을 사용하며 추가 설치는 없습니다. 아래는 이 EXP 디렉터리에서 실행합니다. 경로는 스크립트 위치를 기준으로 해결하므로 다른 작업 디렉터리에서는 스크립트 경로만 지정하면 됩니다. 새 run-id로 재현하면 EXP003의 세 기준 모델과 대응 분기들을 학습하며, EXP001·EXP002는 재학습하지 않습니다. 기존 run-id/seed 결과를 덮어쓰지 않습니다.

```powershell
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$runId = 'reproduction_001'
foreach ($researchSeed in @(42, 123, 2026)) {
    & 'D:\anaconda3\envs\research\python.exe' src\run_seed.py --seed $researchSeed --run-id $runId --device cuda
    if ($LASTEXITCODE -ne 0) { throw 'Training failed; stop before next seed.' }
    & 'D:\anaconda3\envs\research\python.exe' src\verify_seed.py --seed $researchSeed --run-id $runId
    if ($LASTEXITCODE -ne 0) { throw 'Verification failed; stop before next seed.' }
}
& 'D:\anaconda3\envs\research\python.exe' src\analyze.py --run-id $runId --seeds 42 123 2026
```

현재 결과를 재검증할 때는 학습 없이 다음 명령만 실행합니다. 기존 verification.json은 덮어쓰지 않고 재계산 결과가 같은지 비교합니다.

```powershell
& 'D:\anaconda3\envs\research\python.exe' src\verify_seed.py --seed 42 --run-id main_20261022
& 'D:\anaconda3\envs\research\python.exe' src\verify_analysis.py --run-id main_20261022
```

CPU 환경에서도 체크포인트는 `torch.load(path, map_location='cpu', weights_only=False)`로 읽을 수 있습니다. 현재 검증 CLI는 실제 검증 데이터 재평가에 CUDA를 사용합니다. 신뢰할 수 있는 프로젝트 체크포인트만 로드하십시오. 장치·라이브러리 버전이 바뀌면 비트 일치를 보장하지 않습니다.

## 파일 구조와 기록

- `checkpoints/<run_id>/seed_<seed>/baseline/`: step 0, 매 25스텝 전체 상태, completion
- `checkpoints/<run_id>/seed_<seed>/P*/source.pt`: 원래 상태를 정확히 복원한 분기 출발점
- `P*/{control,growth}/`: initial, 첫 업데이트, 매 250스텝, final 체크포인트
- `results/<run_id>/seed_<seed>/baseline/steps.csv`: 매 스텝 train batch loss, lr, gradient norm, 레이어별 업데이트 norm, 샘플 수·배치 위치·시간
- `baseline/validation_coarse.csv`, `completion_refinement.csv`, `validation_exact.csv`: 탐색·구간 재생·전 구간 검증 결과
- `branches/P*/performance.csv`: 매 스텝 train batch loss, 업데이트 후 validation loss/accuracy, 샘플 수·학습 시간·연산 proxy
- `branches/P*/directions.csv`: 기존 좌표만의 task gradient cosine, 실제 업데이트 cosine·norm, 누적 cosine 및 경로 거리
- `branches/P*/new_neuron.csv`: 입력·출력 gradient, 가중치 norm, ReLU 활성, fc2 pre-activation 기여
- `baseline/path/`, `branches/P*/{control_path,growth_path}/`: initial.npy와 25스텝 단위 XOR 청크로 FP32 파라미터 경로를 손실 없이 보존
- `seed_<seed>/{config,environment,data_provenance,summary,verification}.json`, `requirements-freeze.txt`, `source/`: 설정·환경·소스·검증 기록
- `results/<run_id>/analysis/`: 개별 seed/집계 CSV, 과정 그래프 및 Q1~Q5 연구 보고서

각 전체 체크포인트에는 모델, AdamW 상태, Python/NumPy/PyTorch/CUDA RNG, 실제 샘플 순서와 다음 배치 위치, 원본 분할 및 소스 해시가 있습니다. 매 스텝 전체 optimizer 상태를 반복 저장하지 않고, 원본 FP32 경로에서 실제 업데이트를 float64로 정확히 재계산할 수 있게 설계했습니다. 첫 gradient 원본은 보존하고 이후에는 gradient 방향과 norm을 기록합니다. 통합된 33뉴런 행렬과 분리 경로의 수학적 동등성도 검사합니다.

## 지표 해석

원래 weight·bias만 레이어별로 연결합니다. 두 norm 중 하나라도 1e-12 이하면 cosine은 결측입니다. 큰 방향 차이의 관측 기준은 사전에 cosine<0.99로 정했습니다. 이 임계값은 보편적 기준이 아닙니다. 거리의 정규화는 분기 출발 가중치 norm+1e-12를 사용합니다.

Train loss와 신규 뉴런 활성·기여는 업데이트 전 동일 훈련 batch의 값입니다. 검증 지표와 신규 weight norm은 업데이트 후 값입니다. 시간 비교는 매 스텝 실행 순서를 번갈아 측정한 forward/backward 및 optimizer 구간의 동기화 wall time과 CUDA event 시간을 보존합니다. 검증·모니터링·파일 기록을 포함한 전체 계측 시간은 별도입니다.

forward dense-MAC proxy는 Control에서 샘플당 25,760, Growth에서 26,560입니다. 전체 FLOPs가 아니며 계측 없는 전용 시간 benchmark도 아닙니다. 스텝 효율과 계산 효율을 구분하고 동일 시간·proxy 예산의 마지막 실제 관측값으로 비교합니다.

각 seed 내부의 대응 차이를 먼저 계산한 뒤 3개 seed의 평균과 표본 표준편차(ddof=1)를 집계합니다. 서로 다른 모델의 가중치 좌표를 직접 비교하지 않습니다. 방향 차이나 거리가 크다는 이유로 성능 향상을 주장하지 않습니다.

## 완료 상태와 결과

세 seed의 기준 학습, 4시점×Control/Growth=24개 분기 학습, 독립 검증과 집계를 모두 완료했습니다. 기존 EXP001·EXP002의 모든 파일 해시가 동일하며 테스트 데이터는 사용하지 않았습니다.

| seed | S_complete | P25 | P50 | P75 | P100 |
|---|---:|---:|---:|---:|---:|
| 42 | 1,601 | 400 | 800 | 1,200 | 1,601 |
| 123 | 2,257 | 564 | 1,128 | 1,692 | 2,257 |
| 2026 | 1,383 | 345 | 691 | 1,037 | 1,383 |

25스텝 간격 관측만으로는 세 seed 모두 이전 구간의 일시적 도달을 놓쳤습니다. 따라서 전 구간의 가중치 경로를 학습 없이 재평가해 위 최초 도달점을 확정했습니다.

실제 업데이트 cosine<0.99는 36개 seed·시점·레이어 조합 중 35개에서 관측했습니다. 처음 경계를 넘은 추가 step은 72~787입니다. 최초 업데이트의 기존 gradient·실제 업데이트는 동일했고, 이후 task gradient 방향과 optimizer 업데이트 방향이 서로 다른 양상으로 달라졌습니다.

P25는 동일 추가 step 수에서 종료 검증 loss가 세 seed 모두 낮았으며, 평균 정확도 차이는 +0.2267±0.3215%p였습니다. 반면 공통 시간 예산에서는 평균 loss 차이가 +0.00741로 Growth 쪽이 높았습니다. 계측된 학습 구간은 Growth가 약 32~36% 더 걸렸습니다. 구현·계측 조건에 따른 관찰이며 일반적인 시간 효율 우위를 주장하지 않습니다.

신규 출력 gradient는 모든 분기에서 추가 step 1, 입력 gradient는 step 2부터 비영이었습니다. 일부 종료 batch에서는 활성·기여가 0이었으므로, 초기 참여가 관측 기간 내내 지속됐다고 단정하지 않았습니다.

독립 검증에서 232개 분기 체크포인트, 83,856개 스텝·레이어 방향/거리 항목을 확인했습니다. 추가로 최종 집계의 2,564개 artifact 해시와 대응 종료값·예산·도달 step·표본 표준편차를 검사했습니다. 중첩되는 무확장 Control 경로도 비트 단위로 일치했습니다.

- [Q1~Q5 연구 보고서](results/main_20261022/analysis/REPORT.md)
- [개별 seed별 대응 지표](results/main_20261022/analysis/paired_seed_metrics.csv)
- [평균·표준편차](results/main_20261022/analysis/paired_aggregate.csv)
- [목표 도달 step](results/main_20261022/analysis/threshold_steps.csv)
- [스텝·시간·연산 proxy 비교 그래프](results/main_20261022/analysis/figures/06_efficiency.png)
- [방향 변화 그래프 예시](results/main_20261022/analysis/figures/01_directions_seed_123.png)
- [최종 검증 결과](results/main_20261022/final_verification.json)

seed별 summary.json의 상태는 학습 단계 종료 당시의 기록이며, 전체 분석 완료 상태는 analysis/summary.json을 기준으로 확인합니다. 3개 seed의 기술 통계이며 모집단 일반화나 뉴런 추가의 우수성 증명으로 해석하지 않습니다.
