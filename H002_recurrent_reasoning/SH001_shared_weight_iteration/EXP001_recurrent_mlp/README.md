# EXP001 — 공유 가중치 MLP 반복 학습

동일한 64→128→64 MLP(Linear, ReLU, Linear, Sigmoid; 16,576개 파라미터)를 반복 적용하여 8×8 구조적 이진 패턴을 복원합니다. Residual, Attention, Normalization, 외부 메모리는 사용하지 않습니다.

## 실행

이 EXP 디렉터리에서 PowerShell로 실행합니다. 기존 Anaconda 환경을 사용하며 사용자 패키지는 차단합니다.

```powershell
$env:PYTHONNOUSERSITE='1'
& D:\anaconda3\envs\research\python.exe -u src/train.py
```

한 명령으로 3 seed×3 방식 훈련, 테스트 1~32단계 평가, 지연 측정, 복원 검증, 그래프와 한국어 보고서를 생성합니다. 실행 ID에는 시간과 UUID가 포함되며 기존 실행은 덮어쓰지 않습니다. CUDA를 기본 사용하고 지원하지 않으면 PyTorch/CUDA 빌드 정보를 포함한 오류를 냅니다. 의도적인 CPU 실행은 `--device cpu`, 작은 파이프라인 검사는 `--smoke`를 사용합니다. smoke 결과는 본 실험과 구분합니다.

재검증 및 시각화:

```powershell
& D:\anaconda3\envs\research\python.exe src/verify.py results/<run_id>
& D:\anaconda3\envs\research\python.exe src/visualize.py results/<run_id>
```

## 실험 조건

- A: 1단계 BCE, B: 8단계 최종 BCE, C: 8단계 BCE 평균. B/C는 detach 없는 전체 BPTT를 사용합니다.
- seed 42/123/2026, seed별 동일 초기 가중치와 epoch별 동일 배치 순서. 초기 가중치 SHA-256을 기록합니다.
- AdamW, lr=0.001, weight_decay=0.01(PyTorch 기본값 명시), batch=128, 80epoch, FP32, TF32 비활성, deterministic algorithms.
- 각 모델 2,560 optimizer steps. B/C는 A의 8배 mapping 호출을 수행합니다. 동일 연산 예산 실험은 아닙니다.
- 원본: 직선·직사각형 1~4개 조합의 좌우 대칭 패턴. 원본 seed=42, 노이즈 seed=1042/2042/3042, 독립 픽셀 반전 확률=0.15.
- 서로 다른 원본 6,144개를 train=4096/val=1024/test=1024로 분리합니다. 공용 MNIST는 사용하거나 변경하지 않습니다. 고정 데이터와 seed를 실행별로 보존합니다.
- 마지막 epoch 체크포인트를 평가합니다. 검증은 관찰용이며 조기 종료·체크포인트 선택에 사용하지 않습니다. 테스트로 설정을 고르지 않습니다.

## 지표 및 검증

매 단계 BCE, 픽셀 오류·정확도, 전체 패턴 정확 복원율, 직전 대비 개선·악화·정체 비율, 평균·최대 절대 상태 변화, nonfinite·범위 이탈·period-2 비율을 기록합니다. 1단계의 직전 상태는 손상 입력입니다. 진동 검출 규칙과 한계는 자동 보고서에 명시됩니다.

지연 측정은 테스트 앞 128개 고정 배치에서 warm-up 10회 후 30회 측정합니다. model-only는 입력이 이미 장치에 있고, end-to-end는 CPU→GPU 입력과 GPU→CPU 마지막 출력 전송을 포함합니다. CUDA 동기화 후 wall time의 평균과 표준편차를 기록합니다. CPU 실행에서는 GPU 수치로 해석할 수 없습니다.

`verify.py`는 공유 파라미터와 모든 중간 상태의 gradient, 64차원 입출력, 추론 중 가중치 불변, 데이터 분리 및 seed 재생성, 체크포인트 재평가 출력 재현, 기존 H001 및 공용 데이터 전체 파일 해시의 보존을 검사합니다. 실패 시 알고리즘을 자동 교체하지 않고 학습 실패 체크포인트와 이력을 보존합니다.

## 결과 위치와 해석

2026-10-23에 기본 설정의 9개 실행과 전체 검증을 완료했습니다. [측정 결과 및 해석](REPORT.md)을 참고하십시오. A/C는 평균적으로 1회보다 8회·32회 오류가 높았고, B는 2개 seed에서 홀수·짝수 진동을 보였습니다. 반복 학습의 성능 우위는 확인되지 않았습니다.

`results/<run_id>/REPORT.md`에 실제 측정, seed 평균·표준편차, 관측과 미확정 원인을 구분한 한국어 보고서를 저장합니다. 체크포인트는 `checkpoints/<run_id>/`에 저장합니다. 기존 `.gitignore`에 따라 실행 결과·체크포인트는 로컬에 보존되고 Git에 자동 포함되지 않습니다.

원본/손상 입력/1·2·4·8·16·32회 확률 이미지를 동일한 샘플로 비교합니다. 첫 seed의 방식별 1→32회 개선·정체·악화에 해당하는 첫 샘플을 선택하여 합집합을 모든 모델에 공통 사용합니다. 특정 범주가 없으면 `example_selection.json`에 null을 기록합니다.

반복 내 개선, 반복 학습 효과, 계산 효율, 학습 범위를 넘은 안정성을 구분합니다. 반복이 항상 개선된다고 전제하지 않으며, 이 복원 실험이 지속적 사고 능력을 입증하지는 않습니다.
