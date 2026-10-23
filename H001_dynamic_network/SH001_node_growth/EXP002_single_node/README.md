# EXP002 — Single Node Growth

뉴런 1개 추가가 기존 가중치의 학습 궤적에 미치는 영향을 대응 대조군과 비교합니다. **2026-10-22에 네 분기를 실행하고 결과 검증을 완료했습니다.**

## 설정과 기준 모델

[configs/sources.json](configs/sources.json)이 EXP 기준 상대 경로로 EXP001 원본을 참조합니다. 원본 체크포인트는 변경하지 않았습니다. 분기 직전 상태는 기록 요구에 따라 자체 스냅샷으로 저장했습니다.

| 분기 | 출발 시점 | 구조 | 추가 학습 |
|---|---|---|---|
| B0 | EXP001 Epoch 0 (E_mid) | 784→256→128→10 | 1에포크·430스텝 |
| B | Epoch 0 | 784→257→128→10 | 동일 |
| C0 | EXP001 Epoch 1 (E_complete) | 기존 구조 | 동일 |
| C | Epoch 1 | 뉴런 1개 추가 | 동일 |

MNIST 55,000/5,000 고정 분할, 공용 `../../../data/`, 정규화 `(pixel/255-0.1307)/0.3081`, FP32, seed=42입니다. 대응 분기쌍은 원본의 shuffle generator 상태에서 만든 저장 순열을 공유합니다. 모든 분기는 새 AdamW(lr=0.001, weight_decay=0.0001, betas=(0.9,0.999), eps=1e-8)를 사용하며 원본 optimizer moment를 이식하지 않습니다. optimizer 정책은 `experiment_io.py`로 분리했고 이식 정책은 명시적으로 거부합니다.

뉴런은 기존 인덱스 뒤에 정확히 1개 추가합니다. 입력·편향은 seed=42001, 출력 연결은 0입니다. 전체 검증 입력에서 출력 보존(atol=rtol=1e-5)을 선행 검사하고 실패하면 학습하지 않습니다. 상세 설정은 [configs/experiment.json](configs/experiment.json)에 있습니다. 분기 학습은 95%에 도달해도 조기 종료하지 않습니다.

## 실행·재현

기존 `D:\anaconda3\envs\research`를 사용하며 추가 설치는 없었습니다. 아래 명령은 이 EXP 디렉터리에서 실행합니다. 다른 작업 디렉터리에서는 스크립트 경로를 지정하면 됩니다.

```powershell
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\anaconda3\envs\research\python.exe' src\inspect_sources.py
& 'D:\anaconda3\envs\research\python.exe' src\verify_results.py --run-id seed42_20261022
```

EXP001을 재학습하지 않고 네 분기를 재현하려면 새 run_id로 실행합니다. 다음 명령은 EXP002의 추가 학습을 수행하며 기존 run_id 덮어쓰기는 거부합니다.

```powershell
& 'D:\anaconda3\envs\research\python.exe' src\run_experiment.py --run-id reproduction_001 --device cuda
& 'D:\anaconda3\envs\research\python.exe' src\verify_results.py --run-id reproduction_001 --write-audit
```

결정적 GPU 연산을 사용하며 장치·버전 간 비트 일치는 보장하지 않습니다. runner는 원본 검증 logits 해시의 정확한 재현을 전제 검사로 요구하므로 다른 환경에서 실패하면 기록을 변경하거나 자동 재학습하지 않습니다. GPU 없이 체크포인트를 읽을 때는 `torch.load(path, map_location='cpu', weights_only=False)` 또는 `experiment_io.load_checkpoint(path)`를 사용합니다. `verify_results.py`의 검증 재평가는 CUDA를 사용합니다. 이 프로젝트의 신뢰할 수 있는 체크포인트만 로드하십시오.

## 산출물

실행 ID는 `seed42_20261022`입니다.

- `checkpoints/{B0_mid_control,B_mid_growth,C0_final_control,C_final_growth}/<run_id>/`: 각 분기 before_insertion, after_insertion, first_update, epoch_001 전체 상태
- `results/metrics/<run_id>/performance.csv`: 단계별 훈련·검증 손실/정확도; `*_batch_order.npz`: 대응 순열
- `results/weights/<run_id>/parameter_changes.csv`: 기존/신규 텐서와 레이어별 절대·상대 L2, 평균/최대 절대 변화, 분위수
- `paired_changes.csv`: 기존 좌표의 Δ_growth−Δ_control, norm 차이, 업데이트 cosine
- 동일 weights 디렉터리 NPZ: 전체 Δ·단위 방향·히스토그램과 대응 차이 벡터
- `*_first_backward_gradients.pt`, `first_gradient_analysis.json`: 첫 gradient 및 decay 분석
- `results/figures/<run_id>/`: A 정확도, B 기존 상대 변화, C 대응 궤적 차이, D 신규 뉴런, E 레이어 영향, F 변화 분포
- `results/reports/<run_id>/`: 연구 보고서, 요약, 기준 검증, 원본 무결성 기록, 소스·설정 스냅샷, artifact 해시, 독립 검증 결과

체크포인트에는 모델/optimizer/RNG, 원본 출처·해시, 분할, 설정, 분기/누적 step과 성능이 있습니다. 최초 업데이트의 `local_epoch=0`은 미완료 에포크이며 `branch_step=1`로 구분합니다. 순열과 branch_step으로 배치 위치를 식별할 수 있으나 자동 재개 CLI는 이번 범위에 포함하지 않았습니다.

## 확인된 결과

| 분기 | 최종 검증 정확도 | 최종 검증 손실 |
|---|---:|---:|
| B0 | 95.04% | 0.161589320 |
| B | 94.74% | 0.161161738 |
| C0 | 96.70% | 0.109872076 |
| C | 96.64% | 0.109908834 |

추가 직후 최대 logits 차이는 B 출발점에서 8.94e-8, C 출발점에서 1.91e-6으로 허용오차 내였습니다. 기존 좌표는 정확히 보존했습니다. 종료 시 대조군과의 궤적 차이는 절대 L2 기준 두 분기쌍 모두 첫 은닉층에서 가장 컸습니다. 이번 실행에서 확장군의 검증 정확도가 더 높지는 않았습니다.

신규 입력의 첫 gradient는 정확히 0이었고 첫 입력 weight 변화는 6.6551e-8 L2로 AdamW decay와 FP32 반올림의 영향을 관찰했습니다. 신규 출력 gradient는 0이 아니었습니다. 출력의 기준 norm=0이므로 상대 변화보다 절대 변화로 해석합니다.

16개 체크포인트 CPU 로드·검증 출력 재현, 독립 NumPy 변화량 178개와 분기쌍 비교 60개를 통과했습니다. B0의 종료 가중치는 EXP001 완료 모델과 정확히 같았습니다. EXP001 전체 파일 해시도 실행 전후 동일합니다. EXP001 재학습과 테스트 평가는 하지 않았습니다.

E_mid=0, K=1, 단일 seed, 새 optimizer 조건입니다. 일반적 학습 원리나 뉴런 추가의 우수성을 주장하지 않습니다. 상세 해석은 [연구 보고서](results/reports/seed42_20261022/REPORT.md), 검사는 [verification.json](results/reports/seed42_20261022/verification.json)을 참조하십시오.
