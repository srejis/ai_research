# EXP004 — Moving AI 공식 S→G 경로 탐색

공식 MAPF 지도·시나리오를 단일 에이전트4방향 경로 탐색으로 사용합니다. 연결 영역 IoU가 아니라 **실제로 S에서 G까지 도착한 비율**이 주요 지표입니다. 공식 원본과 좌표를 유지하며 모델 실패를 BFS/A*로 복구하지 않습니다.

본 실험 전 [공식 데이터 감사](DATA_AUDIT.md)를 완료했습니다. 동일 지도와 처음 보는 지도의 평가를 분리합니다.

## 실행

이 EXP004 디렉터리에서 PowerShell로 실행합니다.

```powershell
$env:PYTHONNOUSERSITE='1'
& D:\anaconda3\envs\research\python.exe src/prepare_dataset.py
& D:\anaconda3\envs\research\python.exe -u src/train.py --smoke
& D:\anaconda3\envs\research\python.exe -u src/train.py
```

첫 명령은 학습 없는 독립 데이터 감사입니다. 두 번째는 처음128개 학습·32개 검증·각 테스트32개, seed42/2epoch의 파이프라인 검사입니다. 마지막 한 명령으로 고정 공식 데이터에서3seed 학습·실제 경로 평가·기록 교란·복원 검증·그림·보고서를 생성합니다. 고유 run_id를 쓰고 smoke와 본 실험을 섞지 않습니다. 실패는 failure.json 및 가능한 체크포인트/이력으로 보존하며 자동 삭제·조건 변경하지 않습니다. CUDA 기본, 의도적인 CPU만 `--device cpu`를 사용합니다. CUDA 미지원은 빌드 정보와 함께 오류를 냅니다.

```powershell
& D:\anaconda3\envs\research\python.exe src/verify.py results/<run_id>
& D:\anaconda3\envs\research\python.exe src/visualize.py results/<run_id>
```

재검증은 verification_recheck.json에 기록합니다. 시각화 재생성은 원래 예측과 경로만 사용하며 모델을 재학습하지 않습니다.

## 공식 원본·분할

원본15개 ZIP은 루트 `data/movingai_exp004/raw/`에 보존하고 archive/member의 URL·UTC시각·SHA-256을 manifest에 기록합니다. 다운로드는 [Moving AI MAPF 공식 페이지](https://movingai.com/benchmarks/mapf/index.html)의 선택5개 지도 및 even/random 시나리오 링크를 사용합니다. 인용·라이선스는 [공식 안내](https://movingai.com/benchmarks/mapf.html)와 DATA_AUDIT.md에 명시했습니다.

실제9필드 구조와 원본 대각 이동 비용을 검사한 뒤, 4방향 BFS와 독립 Dijkstra로 정답을 다시 만들었습니다. S/G는 원본 시나리오의 x/y만 사용합니다. 지도 문자열G/S는 지형이지 시작/목표가 아닙니다. 지도32×32를 회전·변형·확대하지 않습니다.

67,765행 중 최소12칸·S≠G·중복 제거를 통과한52,516문제를 모두 사용합니다. 학습25,548/검증3,208/동일지도3,227/미관찰지도20,533입니다. 같은 지도·G는 동일한 전체 거리장 정답이므로 목표 지점별로 분리합니다. 미관찰 지도는 maze-32-32-4 및 random-32-32-20입니다. 훈련 거리 기준21/31/46에서 Q1~Q4를 정했습니다. 지도별·구간별 표본 수와 제외 사유는 DATA_AUDIT.md 및 준비 데이터audit.json/row_ledger.jsonl에 있습니다.

## 모델과 목표

| 방식 | 구조 | 사용 정보 |
|---|---|---|
| A | 3072→256→1024 | W/S/G 한 번 |
| A_large | 3072→1088→1024 | 코어+헤드 파라미터 대응 |
| A_flops | 3072→3328→1024 | 코어8회+헤드1회 선형 FLOPs 대응 |
| B | 코어4096→256→1024 공유8회 | W/S/G와 실제 직전R, R8 출력 |
| C | 동결 B + 헤드11264→256→1024 | 원본과 R8복제8개 |
| D | C와 동일 코어·헤드 | 원본과 R1~R8 실제 기록 |
| E | 동일 헤드만 실행 | 원본과0슬롯8개 |

중간 ReLU·최종 Softplus로 정규화 거리 점수를 출력합니다. 코어1,312,000개, 헤드3,147,008개, 합계4,459,008개입니다. A_large는4,458,560개(약0.010% 차이), A_flops는13,635,840개이며 C/D와 선형 forward FLOPs27,262,976/sample이 같습니다. 같은 실제 시간 예산이라는 뜻은 아닙니다. E의0슬롯은 손실에 연결되지 않고 C의 복제 슬롯도 기능적으로 축약 가능하므로 명목 파라미터와 유효 역할이 동일하다고 주장하지 않습니다.

목표는 모든 모델에서 최종 G까지의4방향 거리장입니다. y=d/128, 역변환 d=score×128이며 거리값을 자르지 않습니다. R0는 전부1(128칸 추정)입니다. 벽·G와 분리된 빈칸은 거리 손실에서 제외하며, 벽 지도와 reachable mask로 구분합니다. 도달 가능한 셀만 표본별 평균 정규화 MSE를 계산합니다. 손실 제외 셀의 출력은 임의로 정답 값으로 채우지 않고 코어 상태에도 그대로 남습니다. 모델에 정답 도달 가능 mask를 입력하지 않습니다.

코어는 모든 반복에서 같은 최종 거리장을 감독합니다. loss=0.5×R8 loss+0.5×R1~R8 loss 평균으로 마지막 총 가중치는0.5625입니다. 전체 BPTT, 실제 예측 R 재입력, W/S/G 원본 보존이며 teacher forcing은 없습니다. 코어를 먼저 동결하고 C/D/E 헤드만 최종 MSE로 독립 학습합니다. C/D/E는 같은 초기 헤드·배치 순서를 사용합니다. A와 헤드들도 같은 데이터·최종 거리 목표를 사용합니다. 헤드 결과는 코어에 되돌리지 않습니다.

기본 seed42/123/2026, 코어·헤드·A 각각80epoch, AdamW lr0.001/weight_decay0.01, batch128, FP32, TF32 비활성/deterministic입니다. 마지막epoch를 평가하고 validation/test로 checkpoint를 선택하지 않습니다. End-to-End 재학습과16/32개 기록 확장은 하지 않습니다.

## 공통 경로 디코더

S에서 시작해 합법적인 상하좌우 이웃 중 현재 점수보다 **엄격하게 낮은** 최저 점수 셀을 선택합니다. 동률은 위/아래/왼쪽/오른쪽 순서입니다. G 도착만 성공이며 낮은 이웃 없음·재방문·1,024이동 초과·비유한 값은 실패입니다. G의 예측을 강제로0으로 바꾸거나 탐색·백트래킹·우회 복구를 하지 않습니다. 정답 거리장은 오직 감독/평가/oracle 검사와 정답 그림에만 사용합니다.

모든 모델에 정확히 같은 디코더를 사용합니다. 벽과 범위를 미리 제외하므로 실제 경로 합법성100%는 모델이 합법성을 배웠다는 증거가 아닙니다. 각 결정에서 지도 내4이웃의 최소 원시 점수가 벽인 빈도를 별도 기록합니다. 범위 밖에는 예측 점수가 없어 순위화하지 않습니다. 엄격한 점수 감소가 루프를 구조적으로 막는 점도 구분합니다.

## 평가·교란·비용

성공률, 전체 최적 성공률, 성공 조건부 비용비, 실패0을 포함한SPL, 루프/정체/이동 제한 실패, 도달 셀 거리MAE/RMSE·S 오차·G 점수, 불법 이웃 선호를 저장합니다. 비용비만 성공 사례에서 평균내 전체 성능인 것처럼 보고하지 않습니다. 같은 지도와 미관찰 지도·거리구간·지도별 지표를 따로 제공합니다.

학습된 D에 전체 기록0, R8복제, 역순, 고정 무작위 순열, 다른 공식 문제 기록, 각R1~R8 제거를 적용합니다. 교란은 분포 밖 입력일 수 있으므로 정상 학습한 C−D 비교와 함께 해석합니다.

CUDA warm-up10회 후30회 동기화 wall time을 측정합니다. score GPU, 입력/출력 전송 포함, score+전송+공통CPU 디코더 시간을 분리하고 표준편차를 기록합니다. C/D는 실제로 코어8회+헤드1회를 재계산하고 E는 코어를 실행하지 않습니다. 코어/헤드/전체 선형 FLOPs, optimizer step·호출 수·학습 시간·peak allocated 및 증분·기록 byte 수를 보존합니다. C/D 시스템 비용은 동일 코어 사전학습을 각각 포함하고 캐시 준비/IO는 별도입니다. 선형 FLOPs는 multiply-add=2이며 비선형·mask·cat·역전파·optimizer·디코더는 제외합니다.

## 원본·입력·그림 교차 검증

원본 SHA, 실제 포맷, 모든 공식 좌표와 채택 목록, BFS/Dijkstra 모든 거리장, 적격 전체 oracle 디코더의 합법·최적 성공, 문제/목표 분할과 heldout, W/S/G 입력·공유 코어·모든 상태 gradient·동결·헤드 초기화·배치, 전체 checkpoint/경로/교란 재현을 검사합니다. 선택 그림의 raw map 해시·S/G 좌표·oracle 경로와 Matplotlib marker/line 좌표를 대조하고 visualization_payload.json 및 별도 검증 결과를 남깁니다.

H001·H002 EXP001~003·기존 공용 data의 파일 목록·해시를 보존 검사합니다. 이번에 허용된 새 `data/movingai_exp004/`는 별도 원본/준비 데이터 해시로 검증합니다. 결과·checkpoint·공식 데이터는 기존 ignore 정책을 따릅니다. 원본 source/와 해시, 경로 JSON, 거리장 NPY, 테스트 기록, 지표, 그림, 한국어 보고서를 고유run에 저장하고 Git 관리용 요약REPORT.md를 별도 작성합니다.

정식 모델 평가 점수·다중 에이전트 해결·임의 새 지도에 대한 일반화를 주장하지 않습니다. 전역 거리 MSE가 낮아도 G까지 점수가 매칸 감소하는 것은 보장되지 않습니다. 실패는 실패로 보존합니다.

## 실행 폴더 구분

본 실험 완료 결과는 [한국어 보고서](REPORT.md), [전체 집계·비용·검증 JSON](SUMMARY.json), [대표 그림과 출처 해시](figures/provenance.json)에 보존했습니다. 같은 지도 D 성공률은5.010%, C는4.132%였지만 R8복제 교란도5.123%여서 전체 과거 기록의 필수성은 입증하지 못했습니다. 미관찰 지도 성능은 모든 모델에서 매우 낮았습니다.

| run_id | 용도 |
|---|---|
| `20261024_000032_smoke_2a9f7ac8` | 공식 데이터 부분집합2epoch, 전체 파이프라인·oracle·원본/그림 일치 검증 통과. 본 성능에서 제외 |
| `20261024_001323_main_ff5cf824` | 공식 적격 데이터 전체, 기본3seed 본 실험 |

각 폴더 안에 여러 모델을 함께 보존합니다. checkpoint와results의 같은 run_id는 한 실행의 서로 다른 산출물 위치입니다.

## 중단된 실행 재개와 추가 분석

프로세스가 종료된 것을 확인한 뒤 같은 run을 재개합니다.

```powershell
$env:PYTHONNOUSERSITE='1'
& D:\anaconda3\envs\research\python.exe src/train.py --resume results/20261024_001323_main_ff5cf824
& D:\anaconda3\envs\research\python.exe src/analyze_results.py results/20261024_001323_main_ff5cf824
```

완료 checkpoint와 저장된 평가는 재사용하며, checkpoint를 끝까지 저장하지 못한 단계만 처음부터 실행합니다. 완료된 run은 재개하지 않습니다. 재개 시 기존 연구·준비 데이터 해시와 설정을 검사합니다. 원래 source/는 보존하고 재개 코드·해시·사유는 resume_source_*에 기록합니다. 본 실험은 첫 seed의 코어/C 평가 완료 뒤 세션 중단으로 재개했으며 새 본 실험 폴더를 추가하지 않았습니다. 로그는 본 run의 resume_stdout.log/resume_stderr.log에 있습니다.

analyze_results.py는 저장된 실제 경로와 지표를 읽어 Manhattan 거리(훈련 사분위), 우회 비율, 정답 최단 경로의 꺾임 구간별 성공률, C/D의 같은 문제별 성공 차이와 실패 끝점에서 목표까지 남은 거리를 기록합니다. 꺾임 수는 상/하/좌/우 동률 규칙으로 택한 정답 경로의 속성이며 최소 꺾임 수가 아닙니다. 모델 경로 생성에 이 정답을 사용하지 않습니다. 빈 구간은 표본0과 null로 보존하며 보간하지 않습니다.
