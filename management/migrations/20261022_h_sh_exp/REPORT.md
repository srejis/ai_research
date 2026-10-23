# 디렉터리 이전 및 복원 검증 보고서

작업일: 2026-10-22 (Asia/Seoul). 기존 `research` Anaconda 환경만 사용했으며 추가 설치, 학습, 뉴런 추가, 테스트 데이터 재평가는 수행하지 않았습니다.

## 파일 분류와 이전

이전 전 43개 파일, 합계 71,581,629바이트를 확인했습니다. [before.json](before.json)에 원래 경로·크기·SHA-256, [mapping.json](mapping.json)에 이전 후 대응 경로를 기록했습니다. [move_files.py](move_files.py)는 덮어쓰기를 거부하고 워크스페이스 안의 경로만 이전했습니다. 빈 옛 디렉터리만 제거했습니다.

- 기준 모델 코드, 설정, 환경 설치 내역 → EXP001
- 기존 체크포인트 2개 → `EXP001/checkpoints/mnist_baseline_seed42_20261022/`
- 원본 결과, 로그, 그래프, 소스 스냅샷 → `EXP001/results/`
- 기존 루트 README → `EXP001/results/mnist_baseline_seed42_20261022/legacy_root_README.md`
- 현재 후속 실험 helper → `EXP002/src/future_expansion.py`
- 공용 MNIST와 고정 분할 `data/`, 기존 `AGENTS.md` → 위치와 내용 유지
- 기존 Python 캐시도 삭제하지 않고 EXP001 src와 함께 보존

실험 관련 여부가 불분명한 파일은 발견되지 않았습니다. AGENTS.md는 공통 지침으로 유지했습니다. 실제 프로젝트 폴더명 `00_ai_research`는 변경하지 않았습니다.

## 무결성과 수정 범위

이동 직후 원래 43개 파일 모두 크기·SHA-256이 일치했습니다([after_move.json](after_move.json)). 이후 수정된 기존 파일은 현재 실행용 `train_baseline.py`, `verify_baseline.py`, `future_expansion.py`, 공통 `.gitignore`의 4개뿐입니다. 나머지 원래 39개 파일은 최종 검사에서도 바이트 단위로 동일합니다.

학습 코드 수정은 공용 데이터, EXP 기준 설정/결과 경로, 분리된 체크포인트 위치와 해시 목록 대응입니다. 모델과 학습 알고리즘은 변경하지 않았습니다. 복원 코드는 현재 위치 해석 및 모델·옵티마이저 일치 검사를 추가했습니다. EXP002 helper는 EXP001 모델 import 경로만 수정했습니다. `.gitignore`에는 체크포인트 디렉터리를 추가했습니다. 신규 경로 모듈·참조 설정·검사 코드·문서는 별도로 생성했습니다.

원본 결과와 소스 스냅샷을 수정하지 않았으므로 내부의 옛 절대 경로와 실행 지시는 역사적 기록입니다. 현재 실행은 각 EXP README 및 EXP001 `src/paths.py`를 사용합니다. 원본 artifact manifest의 `checkpoints/` 항목은 검증 코드가 분리된 실제 위치에 대응시킵니다. 원본 해시 목록 자체는 변경하지 않았습니다.

최종 파일별 비교 및 전체 파일 수·크기 목록은 [final_audit.json](final_audit.json)에 있습니다. [audit_preservation.py](audit_preservation.py)로 다시 검사할 수 있습니다.

## 복원 결과

[restoration.json](restoration.json)에 새 경로에서 수행한 검사 결과가 있습니다.

| 시점 | epoch | 검증 정확도 | 검증 손실 | 결과 |
|---|---:|---:|---:|---|
| E_mid | 0 | 5.50% | 2.3032704097747803 | 원본 출력 해시까지 일치 |
| E_complete | 1 | 95.04% | 0.16158931975364685 | 원본 출력 해시까지 일치 |

두 파일의 SHA-256이 원본과 동일하고 모델 strict load, 가중치 전부 일치, AdamW 상태 일치를 확인했습니다. Epoch 0의 optimizer moment는 원래 비어 있습니다. 고정 분할 55,000/5,000의 중복 없음, 원본 artifact manifest 전체 해시 검사도 통과했습니다. 원본에 기록된 테스트 정확도는 95.55%이며 이번에 재측정한 값은 아닙니다.

EXP002 `inspect_sources.py`가 설정의 상대 경로, 두 체크포인트 해시, E_mid/E_complete/K 및 공용 분할 존재를 확인했습니다. 모델 생성이나 뉴런 추가는 하지 않았습니다. EXP001 학습 CLI는 `--help`로만 확인했습니다. Python 문법과 JSON 구문 검사도 통과했습니다.

추천 커밋 이름: `refactor: organize research into H-SH-EXP hierarchy and verify preserved checkpoints`
