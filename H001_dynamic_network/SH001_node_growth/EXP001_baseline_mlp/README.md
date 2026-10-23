# EXP001 — Baseline MLP

MNIST 기준 모델을 정상 학습하고 Epoch 0부터 매 에포크의 모델·옵티마이저·RNG·지표를 보존한 실험입니다. 기존 학습은 완료되었으며 이번 구조 정리에서는 학습하지 않았습니다.

## 설정과 출처

자체 초기화한 `784 → 256 → 128 → 10` MLP이며 외부 기준 체크포인트는 없습니다. ReLU, CrossEntropyLoss, AdamW(lr=0.001, weight_decay=0.0001), batch size=128, FP32, seed=42입니다. Dropout·BatchNorm은 없습니다. TF32는 꺼져 있고 결정적 연산을 사용합니다. 자세한 설정은 [configs/baseline.json](configs/baseline.json), 설치 내역은 [environment_setup.txt](environment_setup.txt)에 있습니다.

공용 MNIST는 `../../../data/`에 있습니다. 고정 분할 `mnist_split_seed42.pt`로 55,000/5,000을 나누고, 테스트는 별도 10,000개입니다. 입력 정규화는 `(pixel/255 - 0.1307)/0.3081`입니다.

## 실행 방법

아래는 이 EXP 디렉터리에서 실행합니다. 기존 환경을 그대로 사용합니다.

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\anaconda3\envs\research\python.exe' src\verify_baseline.py --run-dir results\mnist_baseline_seed42_20261022
```

위 명령은 저장 결과와 체크포인트를 읽어 검증 데이터만 재평가합니다. 작업 디렉터리와 관계없이 상대 `--run-dir` 및 `--config`는 이 EXP 디렉터리 기준입니다. 절대 경로도 가능합니다.

추후 별도로 승인된 기준 모델 재현 학습 명령은 다음과 같습니다. 이번 정리에서는 실행하지 않았습니다. 새 run_id를 사용해야 하며 기존 결과 폴더 또는 체크포인트 폴더가 있으면 실패합니다.

```powershell
& 'D:\anaconda3\envs\research\python.exe' src\train_baseline.py --config configs\baseline.json --run-dir results\new_run_id
```

## 결과와 보존 경로

실행 ID는 `mnist_baseline_seed42_20261022`입니다.

- `checkpoints/<run_id>/epoch_000.pt`, `epoch_001.pt`: 원본 전체 체크포인트
- `results/<run_id>/metrics.csv`, `accuracy.png`, `loss.png`: 원본 로그 및 그래프
- `results/<run_id>/summary.json`, `report_ko.txt`: 원본 결과 및 보고서
- `results/<run_id>/source/`: 변경하지 않은 원본 소스/문서 스냅샷
- `results/<run_id>/artifact_sha256.json`: 원본 해시 목록. 검증 코드는 `checkpoints/` 키를 현재 체크포인트 위치에 대응시킵니다.
- `results/<run_id>/config.json`, `environment.json`, `requirements-freeze.txt`, `data_manifest.json`, `data_split.pt`: 원본 실행 메타데이터
- `results/*console.log`, `*verification.log`: 기존 콘솔 기록
- `results/<run_id>/legacy_root_README.md`: 구조 변경 전 루트 README 원본

원본 보고서·JSON·로그·체크포인트 내부의 옛 절대 경로는 당시 이력으로 그대로 보존했습니다. 현재 경로는 이 README, `src/paths.py` 및 [이전 매핑](../../../management/migrations/20261022_h_sh_exp/mapping.json)을 사용합니다. 원본 `source/`는 보관용이며 이동 후 실행 코드는 `src/`입니다. 미래 뉴런 추가 helper의 현재 위치는 EXP002이고 원본 소스 스냅샷에는 당시 파일을 그대로 남겼습니다.

## 확인된 상태

최초 95% 도달 시점 `E_complete=1`, `E_mid=0`, 후속 추가 학습량 `K=1`입니다. 완료 훈련 정확도는 95.8473%, 검증 정확도는 95.04%(4,752/5,000), 기존 테스트 정확도는 95.55%(9,555/10,000)입니다. 훈련 손실은 2.30354158에서 0.13372503으로 감소했습니다. 완료는 운영적 정의이며 최적화 수렴을 뜻하지 않습니다.

이동 후 두 체크포인트의 SHA-256, 모델 구조·가중치, AdamW 상태가 일치했고, 복원한 검증 손실·정확도·전체 logits 해시가 원본 기록과 정확히 일치했습니다. Epoch 0는 초기화 상태이므로 optimizer moment가 아직 없는 것이 정상입니다. 자세한 검사 기록은 [restoration.json](../../../management/migrations/20261022_h_sh_exp/restoration.json)에 있습니다.
