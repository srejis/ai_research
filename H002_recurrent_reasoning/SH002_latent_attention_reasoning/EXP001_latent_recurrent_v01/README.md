# H002 / SH002 / EXP001 — 잠재 Attention 반복 추론 v0.1

공식 Moving AI 문제에서 최종 방향 정책만 감독하는 잠재 반복 모델을 연구합니다. 기존 SH001 코드·보고서·데이터는 읽기 전용으로 보존합니다. 원문 지시서는 루트 프롬프트 폴더의20261024_H002_SH002_EXP001_잠재_Attention_반복추론.md입니다.

## 완료 결과

3seed×A/B/C의80epoch 학습과9개 동결 종료 후보(비용0/0.001/0.01)의10epoch 학습·복원 검증을 완료했습니다. [한국어 보고서](REPORT.md), [전체 수치·비용·검증 JSON](SUMMARY.json), [실행 ID 목록](RUNS.json)을 보존했습니다.

| 조건 | 관찰 지도 완주율% ± seed SD | 미관찰 지도 완주율% ± seed SD |
|---|---:|---:|
| A: 독립1회 Attention | 6.67 ± 5.82 | 0.060 ± 0.052 |
| B: 공유16회+기록 | 9.12 ± 7.65 | 0.198 ± 0.232 |
| C: 공유16회, 기록 없음 | 7.52 ± 6.34 | 0.135 ± 0.017 |

B의 평균은 높지만 A보다 높은 시드는1/3, C보다 높은 시드는2/3이었습니다. 절대 완주율과 미관찰 지도 일반화는 낮았으며, 반복·기록의 일관된 우월성을 입증하지 못했습니다. 종료 후보들은 검증·관찰 테스트에서 모두 평균16회·상한 강제종료100%로, 성능 유지형 계산 절약에 실패했습니다.

## 데이터·과제

[Moving AI 공식 MAPF 자료](https://movingai.com/benchmarks/mapf/index.html)에서 파생한 단일 에이전트·4방향·단위 비용 문제입니다. 공식 다중 에이전트 점수나 이전 거리장 회귀 성공률과 같은 평가라고 주장하지 않습니다. 데이터셋ID는3e4d5629b9fe7e71이고 train25,548/val3,208/seen3,227/unseen20,533입니다. 기존 문제ID·지도·S/G·목표별 분할을 그대로 사용하고 공식 자료를 추가 다운로드하거나 수정하지 않습니다.

입력은 셀별W/S/G/정규화 행·열의5개 원본 정보만 사용합니다. 새 라벨은 BFS와 독립 Dijkstra로 검증한 거리에서 정확히1 줄이는 모든 합법 방향의 집합입니다. 벽·목표·도달 불가 셀은 손실에서만 제외하며 모델에 정답 mask를 넣지 않습니다. 셀 손실은 -logsumexp(log_softmax(logits)[정답방향])이며 문제별 감독 셀 평균 후 배치 평균입니다.

실제 경로는 각 셀의 원시4방향 argmax를 그대로 따릅니다. 벽/범위 밖 방향을 미리 제거하거나 대체하지 않습니다. 재방문·불법 이동·비유한 값·1,024이동 제한은 실패이며 정답 탐색으로 복구하지 않습니다.

## 모델·학습

| 조건 | 독립 학습 | 코어 반복 | 과거 기록 |
|---|---|---:|---|
| A | 처음부터 학습 | 1 | 없음 |
| B | 처음부터 학습 | 16 | H1~직전 상태 |
| C | 처음부터 학습 | 16 | 없음, X만 참조 |

모두397,828개 파라미터를 갖는 동일 구조입니다. 칸별5→128→128 인코더와 학습 가능한 행/열 위치 표현으로 X=[N,1024,128]을 만듭니다. 인코더에는 전역 attention이 없습니다. H0는 서로 구분되는 학습 가능한16개128차원 벡터입니다. 코어는 별도의Cross-Attention·Self-Attention·128→512→128 GELU MLP, Pre-LayerNorm·잔차로 구성하고 동일 인스턴스를 반복합니다. 4개 attention head의 폭은32입니다.

Cross-Attention은 현재 H를Q, 읽기 전용X와 생성된 과거H를K/V로 사용합니다. 원본/기록 source embedding, 기록 시간 embedding, slot embedding을 구분합니다. H0는 과거 기록에 넣지 않으며 미래 기록은 없습니다. 입력 K/V와 각 과거 기록 K/V는 같은 forward 안에서 한 번 계산해 재사용합니다. 같은 파라미터로 같은 값을 재계산하는 중복만 줄인 것으로, detach나 BPTT 절단은 없습니다. B의 참조 길이는1,024→1,264로 증가하고 C는1,024로 고정입니다.

출력 헤드는 국소5개 특징에서 만든1024개쿼리와 마지막H만의K/V를 사용하는 작은 attention·공유 MLP로4방향 로짓을 냅니다. 과거 기록이나 정책 출력을 코어에 직접 되먹이지 않습니다. 중간 정답 손실은 없습니다. A/C에서도 동일 파라미터 구조를 유지하므로 기록 전용 embedding 일부는 사용하지 않습니다.

AdamW lr0.0003, weight_decay0.01, gradient clip1.0, FP32·TF32꺼짐·deterministic, seed42/123/2026, epoch상한80입니다. 배치는 사전 FP32 BPTT 측정으로 확정하고 설정에 사유를 기록합니다. 검증은1epoch·5epoch 간격·최종epoch에서 수행하며 완주율 최대→SPL최대→정책 손실 최소→먼저 도달한epoch 순으로 checkpoint를 선택합니다. 테스트는 선택에 쓰지 않습니다. 같은seed의 A/B/C 초기 가중치와 epoch별 문제 순서는 대응시킵니다.

실제 채택 배치는256입니다. 사전 측정에서 배치256은 약8.33GiB·2,537문제/초, 배치768은 약22.81GiB·3,241문제/초였습니다. 최대 처리량만 택하면 epoch당 optimizer 업데이트가100회에서34회로 줄어들므로 배치256을 채택했습니다. 80epoch의 optimizer 업데이트는조건당8,000회입니다. 최초 후보16에서 변경한 이유와 측정 원본은 config 및 results/resource_benchmark*.json에 남겼습니다. 학습률·구조·라벨·목적함수는 변경하지 않았습니다.

## 종료 정책

B를 완전히 동결한 뒤 잠재 평균→LayerNorm128→Linear64→GELU→Linear1 종료 헤드를 학습합니다. Bernoulli 조건부 종료 확률을 사용하고 실제 선택한 종결 상태의 헤드만 한 번 읽습니다. 비용은 terminal policy loss +2×실패 +lambda×T/16, lambda=0/0.001/0.01입니다. 비용 최소화 REINFORCE 손실은 (cost-baseline).detach()×선택 로그확률합이며 부호가 양수입니다. baseline은 행동 선택 전에 고정된 이전 minibatch 비용의EMA(beta0.95)로 현재 행동과 독립입니다. 모든 선택 로그확률·실제 행동·비용·baseline을 JSONL에 남깁니다.

먼저lambda=0 종료 정책을 학습하고, 양의비용 후보들은 같은seed의 선택된0비용 헤드에서 각각 분기합니다. 양의비용 분기의 optimizer는 새로 시작하고 추가 학습 예산을 별도로 기록합니다. 동결 모델의 H1~H16 캐시는 종료 학습 계산 최적화이며 중간 정답 손실이나 모든 종료 손실의 가중 평균을 만들지 않습니다. 캐시 생성 비용은 별도입니다. 실제 추론은 활성 문제만 압축해 계속 계산하고 종료된 문제는 이후 코어를 호출하지 않습니다. T16에서continue를 선택하면 상한 강제 종료, halt를 선택하면 자율 종료로 구분합니다. 추론은 p≥0.5에서 결정적으로 종료합니다.

검증 전체와 각Q1~Q4의 성공률·SPL에서 관측 저하0, 평균T<16, batch1 실제 지연 감소를 모두 만족하는 후보만 채택합니다. 불확실성을 별도 보고하며 이 기준이 통계적 동등성 증명은 아닙니다. 만족하는 후보가 없으면 고정16회를 유지합니다. 공동 미세조정과 비공유 깊은 모델은 별도 선택 실험이며 수행 여부와 이유를 보고합니다.

## 실행·재개

EXP001 디렉터리에서 실행합니다.

```powershell
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& D:\anaconda3\envs\research\python.exe src/verify.py --data-only
& D:\anaconda3\envs\research\python.exe src/verify.py --structure
& D:\anaconda3\envs\research\python.exe src/benchmark.py
& D:\anaconda3\envs\research\python.exe src/train_solver.py --phase0 results/<phase0_id> --overfit
& D:\anaconda3\envs\research\python.exe src/train_solver.py --phase0 results/<phase0_id> --smoke
& D:\anaconda3\envs\research\python.exe src/train_solver.py --phase0 results/<phase0_id>
& D:\anaconda3\envs\research\python.exe src/diagnose.py results/<solver_id>
& D:\anaconda3\envs\research\python.exe src/train_halt.py results/<solver_id>
& D:\anaconda3\envs\research\python.exe src/visualize.py results/<solver_id> results/<halt_id>
```

중단된 프로세스가 종료된 것을 확인한 뒤 재개합니다.

```powershell
& D:\anaconda3\envs\research\python.exe src/train_solver.py --phase0 results/<phase0_id> --resume results/<solver_id>
& D:\anaconda3\envs\research\python.exe src/train_halt.py results/<solver_id> --resume results/<halt_id>
```

재개는 설정·현재 소스·데이터 해시가 같은 경우만 허용합니다. 마지막 완료epoch의 모델·optimizer·Python/NumPy/CPU/CUDA RNG를 복원하며 불완전epoch는 반복합니다. 완료한 조건은 재학습하지 않습니다. 각 run의 source/는 불변 스냅샷이며 결과·checkpoint는 동일ID의 서로 다른 산출물입니다. 새 조건은 새ID를 사용합니다.

## 검증과 해석 범위

Phase0은 raw ZIP/member SHA, 실제 원본map/시나리오좌표, 기존분할·문제ID, 모든3,879거리장의 독립BFS/Dijkstra, 모든52,516 Oracle 최적 완주를 검사합니다. 구조 검사는 공유16회·원본불변·전상태gradient·독립history gradient·헤드1회·중간정답 없음·불법방향실패·첫종료와강제16·행동로그확률·동결을 검사합니다. 기존H001/SH001전체와data전체를 실행 전후 SHA-256으로 비교합니다.

잠재 RMS 변화·2단계 간격 변화·Attention 원본/기록 비중·같은B의1/2/4/8회절단과 기록 교란은 평가 진단입니다. 중간 정책 지도에 학습 손실을 추가하지 않습니다. Attention 크기나 잠재 벡터 그림으로 사고 내용을 해독했다고 주장하지 않습니다. `other` 교란은 같은 평가 배치 내 다른 공식 문제 기록의 순환 대체이며, 역순/순열은 시각 태그에 값을 재배치합니다. 태그까지 함께 단순순서만 바꾸는 무효한 순열과 구분합니다.

실패한 소규모 과적합·OOM·중단도 삭제하지 않습니다. 본 일반화 성능과 과적합 성능, smoke를 합치지 않습니다. 원본출처는 Moving AI이며 정책 라벨은 새로 계산한 파생 감독입니다. 라이선스·인용은 기존 공식 manifest의 Open Data Commons Attribution License 및 Stern et al., SoCS2019를 따릅니다.

## 이번 실행의 폴더 구분

| ID 또는 종류 | 용도 |
|---|---|
| `20261024_161159_phase0` | 공식 데이터·Oracle·기존 파일3,799개 기준 해시 |
| `*_overfit_*` | 같은 공식16문제의 학습 가능성 검사. 2천·1만·2만 상한 연장을 별도 보존 |
| `20261024_163258_smoke_c663a1ae` | solver A/B/C의2epoch·일부 데이터 통합 검사 |
| `20261024_163311_halt_smoke_fa3af49f` | 동결 solver의 종료 학습·복원 통합 검사 |
| `20261024_163337_solver_390e42f9` | 동일 공식 분할의3seed 본 solver 학습 |
| `*_halt_*` | 저장된 본 B에서 분리한 종료 정책 학습. smoke와 구분 |
| `20261024_181350_halt_57bd85b2` | 본3seed의9개 동결 종료 후보 학습 |
| `20261024_185845_runtime_review_33602d1b` | 학습 없이 전체 정책 재현과 다양한 검증 문제의 지연 측정·최종 그림 감사 |

소규모2천스텝에서는 A/B 모두 최종 완주율6.25%로 엄격 기준에 못 미쳤습니다. 같은 구조·라벨·손실로 최적화만 연장한 결과 B는7천스텝에서15/16완주·loss0.0787로 통과했고, A는2만스텝에서8/16완주·loss0.1072로 미달했습니다. 소규모 결과를 일반화 성능으로 취급하지 않으며, 본 결과 해석에 A의 학습 가능성 검사 한계도 포함합니다. 통과한 B는 이후 연장에서 다시 학습하지 않았습니다.

각 phase가 다른 목적이므로 폴더를 분리하고, 같은 run 안의 여러 모델은 함께 보존합니다. 최종 전체 ID 목록과 분류는 RUNS.json, 학습·평가 집계는 SUMMARY.json에 기록합니다.

## 최종 추론 감사 재현

본 학습 당시 종료 판단 로짓을 한 반복 안에서 중복 계산하던 부분을 한 번 계산해 재사용하도록 정리했습니다. 모든9개 정책의 전체 테스트 로짓 해시·경로·지표 및 검증 결과가 정확히 같았습니다. 가중치·라벨·decoder·정책 목적함수는 바꾸지 않았고 재학습도 하지 않았습니다. 원래 소스·측정·보고서와 최종 감사 버전을 각각 보존했습니다.

초기 지연은 첫 문제/배치 반복 측정이었고, 최종 지연은 훈련 기준Q1~Q4로 층화한 여러 공식 검증 문제를 사용해 고정16회/자율종료의 측정 순서를 번갈았습니다. batch1/16/64의 평균·표준편차·관측최대와 문제ID를 기록했습니다. 측정된 최대값은 모든 문제에 대한 최악 지연 보장이 아닙니다.

```powershell
& D:\anaconda3\envs\research\python.exe tools/review_runtime.py results/workflow_20261024_163219.json
& D:\anaconda3\envs\research\python.exe tools/finalize_artifacts.py results/<new_runtime_review_id>
```

finalize_artifacts.py는 미생성 final_figures 폴더에 새 그림을 만들며 기존 감사 결과를 덮어쓰지 않습니다. 완료된 학습 run은 소스가 일치해도 재개하지 않습니다. 코드 버전이 바뀐 뒤 기존 학습을 이어가려면 기존 결과를 보존하고 새 run의 조건을 명시해야 합니다. 기록된 최종 감사 버전과 학습 당시 source/는 서로 구분됩니다.

구현 참고: [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html), [PyTorch REINFORCE 예시](https://docs.pytorch.org/docs/2.14/distributions.html#score-function). 실제 사용 환경은 각run의 environment.json에 기록합니다.
