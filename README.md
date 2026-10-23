# AI Research

여러 가설과 실험을 `H → SH → EXP` 계층으로 관리하는 연구 저장소입니다.

| 상위 연구 | 부분 가설 | 실험 | 상태 |
|---|---|---|---|
| [H001 Dynamic Network](H001_dynamic_network/README.md) | [SH001 Node Growth](H001_dynamic_network/SH001_node_growth/README.md) | [EXP001 Baseline MLP](H001_dynamic_network/SH001_node_growth/EXP001_baseline_mlp/README.md) | 학습 완료, 이동 후 복원 검증 통과 |
| H001 | SH001 | [EXP002 Single Node Growth](H001_dynamic_network/SH001_node_growth/EXP002_single_node/README.md) | 4개 분기 완료, 복원·변화량 검증 통과 |
| H001 | SH001 | [EXP003 Learning Trajectory](H001_dynamic_network/SH001_node_growth/EXP003_learning_trajectory/README.md) | 축소 모델 3seed·12개 분기쌍 실행 및 검증 완료 |

## 공통 관리 규칙

- 신규 연구도 `Hnnn_주제/SHnnn_질문/EXPnnn_실험` 구조를 따릅니다.
- 각 실험은 `src/`, `configs/`, `checkpoints/`, `results/`, `README.md`를 관리합니다.
- `data/`는 공용 MNIST 및 고정 분할 저장소입니다. 실험별로 데이터셋을 복제하지 않습니다. 기존 결과에 포함된 분할 스냅샷은 원본 보존을 위해 유지합니다.
- 실행 경로는 각 EXP 기준입니다. EXP001 체크포인트는 `checkpoints/<run_id>/`, EXP002는 `checkpoints/<branch>/<run_id>/`에 저장하고 각 EXP의 `results/`에서 지표·그래프를 관리합니다.
- 다른 실험의 기준 체크포인트는 설정에서 상대 경로로 참조합니다. 원본 결과와 체크포인트를 수정하거나 덮어쓰지 않습니다.
- 실행 당시의 `results/<run_id>/source/`는 보존용 소스 스냅샷입니다. 현재 실행에는 `EXP/src/` 코드를 사용합니다.
- 기존 Anaconda 환경 `D:\anaconda3\envs\research`를 사용합니다. 환경 폴더는 저장소 밖에 둡니다.
- `AGENTS.md`는 프로젝트 작업 지침, `프롬프트/`는 작업 지시문, `management/`는 구조 변경 감사 기록을 관리합니다.

## 이번 정리 결과

2026-10-22에 기존 43개 파일(71,581,629바이트)을 기록한 뒤 이동했습니다. 이동 직후 전 파일 크기와 SHA-256이 일치했습니다. 이후 현재 실행용 코드 2개에 경로 처리 및 복원 검사를 반영했습니다. 원본 학습 결과, 체크포인트, 데이터, 소스 스냅샷은 변경하지 않았습니다.

실제 확인된 기준 결과는 `E_complete=1`, `E_mid=0`, 검증 정확도 95.04%, 기존 테스트 정확도 95.55%입니다. 디렉터리 이전 단계에서는 학습하지 않았습니다.

상세 내용: [이전 및 검증 보고서](management/migrations/20261022_h_sh_exp/REPORT.md).

## EXP002 실행 결과

2026-10-22 후속 지시에 따라 EXP001을 재검증한 뒤 B0/B/C0/C를 각각 K=1에포크·430스텝 추가 학습했습니다. EXP001의 모든 파일은 보존했습니다. 최종 검증 정확도는 B0 95.04%, B 94.74%, C0 96.70%, C 96.64%입니다. 16개 체크포인트 복원 및 독립 수치 검사를 통과했습니다. 테스트 데이터는 평가하지 않았습니다.

단일 seed이며 E_mid=0은 초기 모델입니다. 뉴런 추가의 일반적 우수성을 뜻하지 않습니다. 상세 내용은 [EXP002 연구 보고서](H001_dynamic_network/SH001_node_growth/EXP002_single_node/results/reports/seed42_20261022/REPORT.md)를 참조하십시오.

## EXP003 실행 결과

축소 모델 784→32→16→10의 최초 검증 95% 도달 step은 seed 42/123/2026에서 각각 1,601/2,257/1,383입니다. 각 seed의 P25/P50/P75/P100에서 뉴런을 1개씩 추가하고, 기존 AdamW 상태를 보존한 Control과 동일한 배치로 비교했습니다. EXP001·EXP002의 모든 파일은 그대로 보존했습니다.

실제 업데이트 방향은 달라졌고, P25에서 같은 추가 step 수의 평균 종료 loss는 낮아졌습니다. 하지만 동일 시간 예산에서의 평균 성능 이득은 확인되지 않았습니다. 스텝 효율과 계산 효율을 구분한 결과는 [EXP003 연구 보고서](H001_dynamic_network/SH001_node_growth/EXP003_learning_trajectory/results/main_20261022/analysis/REPORT.md)에 있습니다.
