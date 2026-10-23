# SH001 — Node Growth

새로운 뉴런을 추가한 직후와 추가 학습 과정에서 기존 가중치가 어떻게 재조정되는지 측정합니다. 원래 뉴런과 연결의 인덱스를 유지하고 새 파라미터와 분리해 추적합니다.

| 실험 | 목적 | 상태 및 확인된 결과 |
|---|---|---|
| [EXP001 — Baseline MLP](EXP001_baseline_mlp/README.md) | MNIST 기준 MLP를 학습하고 매 에포크 전체 상태 저장 | 완료: E_complete=1, E_mid=0, 검증 95.04% |
| [EXP002 — Single Node Growth](EXP002_single_node/README.md) | 중간/완료 시점의 뉴런 1개 추가와 대조군 비교 | 완료: B0 95.04%, B 94.74%, C0 96.70%, C 96.64% (검증 정확도) |
| [EXP003 — Learning Trajectory](EXP003_learning_trajectory/README.md) | 축소 모델에서 추가 시점별 매 스텝 gradient·업데이트 방향 및 학습 효율 비교 | 3seed·12개 분기쌍 완료, AdamW 상태 보존·경로 검증 통과 |

EXP002는 EXP001 원본 체크포인트를 직접 참조하여 네 분기를 K=1에포크씩 실행했습니다. `E_mid=0`은 학습 전 모델입니다. 가중치 변화가 크다는 이유만으로 성능이 개선됐다고 해석하지 않습니다. 상세 비교는 [연구 보고서](EXP002_single_node/results/reports/seed42_20261022/REPORT.md)에 있습니다.

EXP003은 독립 축소 기준 모델을 seed 42/123/2026에서 학습했습니다. 최초 95% 도달 step은 1,601/2,257/1,383으로, 모두 양수인 네 분기점에서 비교했습니다. 매 스텝의 실제 업데이트 방향과 task gradient 방향, loss·정확도·시간·경로 거리·신규 뉴런 상태를 보존했습니다. 상세 결과는 [EXP003 보고서](EXP003_learning_trajectory/results/main_20261022/analysis/REPORT.md)를 참조하십시오.
