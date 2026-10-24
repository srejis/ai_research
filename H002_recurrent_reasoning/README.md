# H002 — Recurrent Reasoning

동일한 가중치를 반복 적용했을 때 상태와 정답 품질이 어떻게 변하는지 연구합니다. 지속적인 사고 능력은 이번 실험의 검증 대상이 아닙니다.

- [SH001 — Shared Weight Iteration](SH001_shared_weight_iteration/README.md)
- [SH002 — Latent Attention Reasoning](SH002_latent_attention_reasoning/README.md)

SH001은 EXP001의 패턴 복원에 이어 [EXP002 미로 전파](SH001_shared_weight_iteration/EXP002_maze_propagation/README.md)에서 원본 입력 보존과 단계별 지도학습의 영향을 비교합니다.

[EXP003 복잡도·기록 통합](SH001_shared_weight_iteration/EXP003_complexity_history_fusion/README.md)은 미로 크기와 BFS 거리별로 동결 반복 코어의 과거 기록을 비순환 출력 MLP에 통합하는 효과를 비교합니다.

[EXP004 공식 경로 탐색](SH001_shared_weight_iteration/EXP004_official_pathfinding/README.md)은 Moving AI 공식 지도·시나리오의 S/G를 그대로 사용하고 실제 도착 성공률과 경로 비용을 측정합니다.

[SH002 EXP001 잠재 Attention v0.1](SH002_latent_attention_reasoning/EXP001_latent_recurrent_v01/README.md)은 자유로운16×128 잠재 상태에서 원본·과거 기록을 읽고 최종 방향 정책만 감독합니다. 독립1회·공유16회·기록 없는16회 및 동결 모델의 학습 가능한 종료를 구분합니다.
