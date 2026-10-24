"""Korean, Git-managed evidence report and aggregate exports; no training decisions."""
import math
import shutil
from pathlib import Path
import numpy as np
from common import BASE,read,write,sha


def meanstd(values):
    return float(np.mean(values)),float(np.std(values,ddof=1 if len(values)>1 else 0))


def uncertainty(k,n):
    # Descriptive Wilson interval only; S/G problems share maps/targets, not independent trials.
    z=1.96; p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,c-r),min(1.,c+r)]


def report(solver_run,halt_run,state,figures_source=None):
    solvers=read(solver_run/'summary.json'); halts=read(halt_run/'summary.json'); config=read(solver_run/'config.json')
    aggregates=[]
    for mode in ('A','B','C'):
        rows=[r for r in solvers if r['mode']==mode]
        for group in rows[0]['groups']:
            item={'mode':mode,'group':group,'n':rows[0]['groups'][group]['n']}
            if item['n']:
                for key in ('success','optimal','spl','loss','wall','out_of_bounds','revisit','high_confidence_wrong','success_cost_ratio','optimal_among_success'):
                    values=[r['groups'][group][key] for r in rows if r['groups'][group][key] is not None]
                    item[key+'_mean'],item[key+'_std']=meanstd(values) if values else (None,None)
            aggregates.append(item)
    selected=[]
    for seed in config['seeds']:
        s=next(r for r in halts if r['selection']['seed']==seed); adopted=s['selection']['adopted_penalty']
        if adopted is None:
            b=next(r for r in solvers if r['mode']=='B' and r['seed']==seed)
            selected.append({'seed':seed,'condition':'fixed16 fallback','groups':b['groups'],'latency':b['latency']})
        else:
            r=next(r for r in s['candidates'] if r['penalty']==adopted)
            selected.append({'seed':seed,'condition':f'adaptive cost{adopted}','groups':r['groups'],'latency':r['latency']})
    snapshot={'solver_run':solver_run.name,'halt_run':halt_run.name,'workflow':state,'config':config,'aggregate':aggregates,
              'solver_seed_results':solvers,'halting':halts,'adopted_or_fallback':selected,
              'diagnostics':read(solver_run/'diagnostic_results.json'),'solver_verification':read(solver_run/'verification.json'),
              'halt_verification':read(halt_run/'verification.json'),'dataset_audit':read(solver_run/'dataset_audit.json'),
              'preflight':read(Path(state['overfit'])/'result.json'),'resource_benchmark':read(BASE/'results/resource_benchmark.json')}
    snapshot['preflight_history']=[{'run_id':p.name,'results':read(p/'result.json'),
            'extension':read(p/'extension.json') if (p/'extension.json').exists() else None}
            for p in sorted((BASE/'results').glob('*_overfit_*')) if (p/'result.json').exists()]
    intervals=[]
    for s in halts:
        for r in s['candidates']:
            n=r['validation']['val']['n']; k=round(r['validation']['val']['success']*n)
            baseline_k=round(r['baseline_validation']['val']['success']*n)
            intervals.append({'seed':r['seed'],'penalty':r['penalty'],'n':n,'candidate_successes':k,
                              'candidate_Wilson95_descriptive':uncertainty(k,n),
                              'baseline_successes':baseline_k,'baseline_Wilson95_descriptive':uncertainty(baseline_k,n),
                              'caveat':'shared map/goal correlation; binomial interval is descriptive, not a valid equivalence or noninferiority proof'})
    snapshot['validation_uncertainty']=intervals
    write(BASE/'SUMMARY.json',snapshot)
    figures=BASE/'figures'; figures.mkdir(exist_ok=True)
    figures_source=Path(figures_source) if figures_source is not None else solver_run/'figures'
    for p in figures_source.iterdir():
        if p.is_file(): shutil.copyfile(p,figures/p.name); assert sha(p)==sha(figures/p.name)
    def get(mode,group): return next(r for r in aggregates if r['mode']==mode and r['group']==group)
    def metric(r,key,percent=False):
        a,b=r.get(key+'_mean'),r.get(key+'_std')
        if a is None: return 'N/A'
        multiplier=100 if percent else 1
        return f'{a*multiplier:.4f} ± {b*multiplier:.4f}'
    text=['# EXP001 — 잠재 Attention 반복 추론 v0.1 결과','',
          f'본 solver run: `{solver_run.name}`, halt run: `{halt_run.name}`. seed={config["seeds"]}. FP32, 모델별{config["epochs"]}epoch, 배치{config["batch_size"]}. Smoke와 과적합은 본 성능에서 제외했습니다.','',
          '## 먼저 확인한 한계','',
          f'16개 공식 훈련 문제의 과적합 엄격 기준(완주율≥90%, 최종loss<0.15)에 A/B 모두 통과했는가: {state["preflight_strict_pass"]}. 조건별 결과는 아래 표에 구분했습니다. 손실 감소와 일부 완주를 확인해도 모든 조건이 소집합을 완전히 학습했다는 뜻이 아닙니다. 일반화 실패를 미로 복잡도만으로 설명하지 않습니다.','',
          '## 실제 완주·최적성','',
          '| 평가 | 모델 | n/seed | 성공률% ± seed SD | SPL | 전체 최적 성공률% | 최종 정책 손실 |',
          '|---|---|---:|---:|---:|---:|---:|']
    preflight_table=['','과적합 검사는 같은 공식16문제·구조·라벨·손실을 유지했고, 학습 스텝만2천→1만→2만 상한으로 연장했습니다. 이미 기준에 도달한 B는 재학습하지 않았습니다. 실패 결과와 연장 출처도 보존합니다.','',
                     '| 모델 | 초기loss | 마지막loss | 최종 완주율 | optimizer 누적steps | 엄격 기준 |','|---|---:|---:|---:|---:|---|']
    for mode,r in snapshot['preflight'].items():
        preflight_table.append(f'| {mode} | {r["initial_loss"]:.6f} | {r["final_loss"]:.6f} | {r["final_success"]*100:.2f}% | {r["steps"]} | {r["strict_pass"]} |')
    pos=text.index('## 실제 완주·최적성'); text[pos:pos]=preflight_table+['']
    for split in ('seen','unseen'):
        for mode in ('A','B','C'):
            r=get(mode,split); text.append(f'| {split} | {mode} | {r["n"]} | {metric(r,"success",True)} | {metric(r,"spl")} | {metric(r,"optimal",True)} | {metric(r,"loss")} |')
    text+=['','A는 독립 학습한1회 모델, B는 기록을 읽는 공유16회, C는 원본만 읽는 공유16회입니다. 같은seed의 초기 구조·가중치·학습 문제 순서를 대응시켰습니다. 중간 정책 손실 없이 마지막 출력만 감독했습니다.','']
    for split in ('seen','unseen'):
        a,b,c=[get(m,split)['success_mean'] for m in ('A','B','C')]
        text.append(f'- {split}: B−A={(b-a)*100:+.4f}%p, B−C={(b-c)*100:+.4f}%p. 절단한 B의1회 추론을 A 기준선으로 사용하지 않았습니다.')
    text+=['','## 거리별·지도별','', '| 평가·구간 | n/seed | A 성공률% | B 성공률% | C 성공률% |','|---|---:|---:|---:|---:|']
    for split in ('seen','unseen'):
        for group in [f'{split}/Q{k}' for k in range(1,5)]+[f'{split}/map{k}' for k in range(5)]:
            r=get('B',group)
            if r['n']: text.append(f'| {group} | {r["n"]} | '+' | '.join(metric(get(m,group),'success',True) for m in ('A','B','C'))+' |')
    text+=['','map0~4는 maze-32-32-2 / maze-32-32-4 / random-32-32-10 / random-32-32-20 / room-32-32-4입니다. Q1=12~21, Q2=22~31, Q3=32~46, Q4≥47입니다. 빈 구간은 JSON에 n=0으로 남기고 보간하지 않았습니다.','',
           '![학습·검증 곡선](figures/training_validation.png)','',
           '![실제 성공률과 난이도](figures/success_by_difficulty.png)','',
           '## 종료 정책: 검증으로 선택, 테스트는 결과 확인만','',
           '| seed | lambda | 검증 성공률% | 검증 평균T | 관찰 성공률% | 미관찰 성공률% | 관찰 강제16 비율 | 품질 유지 | 실제 절약 채택 |','|---|---:|---:|---:|---:|---:|---:|---|---|']
    for s in halts:
        for r in s['candidates']:
            text.append(f'| {r["seed"]} | {r["penalty"]} | {r["validation"]["val"]["success"]*100:.4f} | {r["validation"]["val"]["steps"]:.3f} | {r["groups"]["seen"]["success"]*100:.4f} | {r["groups"]["unseen"]["success"]*100:.4f} | {r["groups"]["seen"]["forced"]:.4f} | {r["quality_qualified"]} | {r["adoptable"]} |')
        text.append(f'\nseed{s["selection"]["seed"]} 최종 선택: `{s["selection"]}`.\n')
    text+=['','검증 전체와 각Q1~Q4의 성공률/SPL 관측 저하0, 평균T 감소, batch1 실측 지연 감소를 모두 요구했습니다. 고정16회 검증 성공이0이면 계산 감소만으로 채택하지 않습니다. 조건을 만족하지 못하면 원래 고정16회 결과를 유지합니다. 강제16회는 마지막Bernoulli continue, 자율16회는 halt입니다. 실제 추론에서는 종료 문제를 활성 배치에서 제거합니다.','',
           'REINFORCE는 실제 선택한 종료 상태 한 번의 정책 손실+2×실패+lambda×T/16 비용을 최소화합니다. 비용baseline은 이전 minibatch 비용EMA로 현재 행동과 독립이며, loss=(cost-baseline).detach()×선택logp합입니다. 모든 가능한 종료 손실을 평균하는 목적함수로 바꾸지 않았습니다. 로그확률·행동·비용·baseline은 각후보actions.jsonl에 있습니다. 동결16회 잠재 캐시는 학습 최적화이며 캐시준비시간과 실제adaptive지연은 분리했습니다.','',
           '![종료 분포와 품질·계산량](figures/stopping_quality.png)','',
           '![난이도별 종료 횟수와 성공](figures/stopping_difficulty.png)','',
           '## 비용','', '| 모델 | 파라미터 | 평균 학습초 | train optimizer steps | 선형·attention forward MFLOPs/문제 | batch1 모델 ms | batch16 ms | batch64 ms |','|---|---:|---:|---:|---:|---:|---:|---:|']
    uncertainty_text=['','검증 성공률의 기술적 Wilson95% 구간(문제 간 상관을 무시한 참고치)은 다음과 같습니다. 통계적 비열등성 증명으로 사용하지 않습니다.','',
                      '| seed | lambda | 고정16회 성공률 구간% | 종료정책 성공률 구간% |','|---|---:|---|---|']
    for r in intervals:
        b=r['baseline_Wilson95_descriptive']; c=r['candidate_Wilson95_descriptive']
        uncertainty_text.append(f'| {r["seed"]} | {r["penalty"]} | {100*b[0]:.3f}~{100*b[1]:.3f} | {100*c[0]:.3f}~{100*c[1]:.3f} |')
    insert=text.index('## 비용'); text[insert:insert]=uncertainty_text+['']
    for mode in ('A','B','C'):
        rs=[r for r in solvers if r['mode']==mode]; first=rs[0]
        text.append(f'| {mode} | {first["parameters"]} | {np.mean([r["own_train_seconds"] for r in rs]):.2f} | {first["optimizer_steps"]} | {first["forward_flops_per_sample"]/1e6:.3f} | '+' | '.join(f'{np.mean([r["latency"][k]["mean_ms"] for r in rs]):.3f}' for k in range(3))+' |')
    text+=['','배치별 std·최대 지연·추가 메모리, 훈련peak allocated/증분, 코어·헤드 호출 횟수, 종료후보별 실제 지연은 SUMMARY.json에 있습니다. FLOPs는 곱셈+덧셈2, Linear 및QK/AV만 포함하고 LayerNorm·GELU·softmax·역전파·optimizer·CPU디코더는 제외합니다. 공유 가중치가 계산량 또는 메모리 일정함을 뜻하지 않습니다. 과거 기록이 늘면 attention 길이도 늘어납니다.','',
           '## 반복 진단과 실패 해석','',
           '![잠재 변화와 원본/기록 Attention 비중](figures/latent_attention.png)','',
           '![동일 B checkpoint의 반복 절단](figures/fixed_iteration_diagnostic.png)','',
           'B의1/2/4/8회 절단, 기록 제거/0/역순/순환 재배치/다른 공식 문제 기록 결과는 SUMMARY.json의 diagnostics에 있습니다. 모두 사후 진단이며 중간 정답 손실을 훈련에 추가하지 않았습니다. 잠재 RMS 변화와2-step delta는 수렴/진동의 수치 진단이지 생각의 의미를 보여주는 증거가 아닙니다. Attention 비중도 사고내용 설명으로 해석하지 않습니다. 교란은 분포 밖 입력일 수 있으므로 정상 학습 B−C가 기록의 직접 비교입니다.','']
    for split in ('seen','unseen'):
        for mode in ('A','B','C'):
            r=get(mode,split)
            text.append(f'- {split}/{mode}: 벽 실패{r["wall_mean"]:.6f}, 범위밖{r["out_of_bounds_mean"]:.6f}, 재방문{r["revisit_mean"]:.6f}, 감독셀의확신≥0.9오답{r["high_confidence_wrong_mean"]:.6f}. 성공조건부 비용비={r["success_cost_ratio_mean"]}, 성공중최적비율={r["optimal_among_success_mean"]}.')
    text+=['','## 데이터·무결성·한계·후속','',
           '[Moving AI 공식 출처](https://movingai.com/benchmarks/mapf/index.html)의 기존 원본ZIP/member SHA와 실제map벽·원본시나리오좌표를 대조했습니다. 데이터셋3e4d5629b9fe7e71의문제ID·분할은EXP004와동일합니다. 3,879거리장을독립BFS/Dijkstra로검증하고52,516전체문제의Oracle최적완주100%를확인했습니다. 출처는Moving AI, 다중정답방향라벨은이번에계산한파생감독입니다. Open Data Commons Attribution License, Stern et al., “Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks”, SoCS2019를인용합니다.','',
           f'기존H001/SH001전체·공식자료를포함한data전체 {read(solver_run/"verification.json")["preserved_files"]}파일의실행전후해시가일치했습니다. 모든solver와종료정책의checkpoint복원후로짓hash·경로·지표가재현되고,종료학습에서solver가중치가불변임을검사했습니다. 원본X불변·공유16회·모든Attentiongradient·별도historygradient·최종헤드1회·행동로그확률·첫종료/강제16·종료후추가코어없음의구조검사를통과했습니다.','',
           'seed표본SD는신뢰구간이나유의성검정이아닙니다. 같은지도·G의여러S가있어문제들이독립이지않고미관찰지도는2개뿐입니다. 엄격한무저하선택도모집단에서정확한동등성을증명하지못합니다. 완주못한모델의생각횟수감소만을자율사고성공으로판정하지않습니다. 기존EXP004는감독·정책디코더가달라그성공률과새구조의차이를잠재상태의순수효과로해석하지않습니다.','',
           '이번기본v0.1은동결solver종료학습까지분리검증했습니다. 공동미세조정은새손실조건이고비공유깊은모델은추가계산예산이므로수행하지않았습니다. 다음연구에서는소규모완전과적합이부족한경우출력병목/최적화를우선분리검증하고한가지조건만변경하는것을권합니다. 충분한고정16회풀이능력과유효한중간readout이확인된뒤공동종료미세조정을별도run에서검토해야합니다.','',
           '## 원본 seed 결과','', '| 모델 | seed | 선택epoch | seen 성공률% | unseen 성공률% |','|---|---:|---:|---:|---:|']
    for r in solvers: text.append(f'| {r["mode"]} | {r["seed"]} | {r["selected_epoch"]} | {r["groups"]["seen"]["success"]*100:.4f} | {r["groups"]["unseen"]["success"]*100:.4f} |')
    text+=['','## 실행과변경','',
           '실행명령은[README](README.md)에있습니다. 구현은공식데이터읽기·다중정답라벨·잠재Attention·solver학습/재개·종료REINFORCE·공통디코더/평가·기록진단·시각화·검증·자동실행/보고서로분리했습니다. 지시문원문을프롬프트에보존하고상위README에SH002링크를추가했습니다. 대용량원본/체크포인트/문제별로짓·경로는기존ignore정책으로로컬에보존하며요약JSON·이보고서·대표그림은Git관리대상입니다.','']
    choices=read(figures_source/'provenance.json')['samples']
    for p in choices[:4]: text.append(f'![실제 공식 문제 {p["record_index"]}](figures/{p["image"]})\n')
    (BASE/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    write(BASE/'RUNS.json',{'workflow':state,'solver':solver_run.name,'halt':halt_run.name,
          'all_run_directories':[p.name for p in sorted((BASE/'results').iterdir()) if p.is_dir()],
          'note':'phase0=reference audit; overfit=training feasibility (including failed/extended); smoke=plumbing only; solver=main; halt=frozen-policy stage; matching checkpoint/result IDs are one run'})
