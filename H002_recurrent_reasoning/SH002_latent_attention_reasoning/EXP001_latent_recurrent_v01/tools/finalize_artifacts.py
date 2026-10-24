"""Documentation and visual QA exports from immutable completed experiment artifacts."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import BASE,ROOT,DATA,read,write,sha,source_hashes
from labels import directions
from evaluate import decode
from dataset import MAPS


def finalize(runtime_run):
    assert read(runtime_run/'complete.json')['all_predictions_unchanged']
    summary=read(BASE/'SUMMARY.json'); solvers=summary['solver_seed_results']; halts=summary['halting']
    assert all(r['validation']['val']['steps']==16 and r['groups']['seen']['steps']==16 and r['groups']['seen']['forced']==1
               for s in halts for r in s['candidates']), 'Review narrative claims if stopping outcomes differ'
    assert all(s['selection']['adopted_penalty'] is None for s in halts)
    solver_run=BASE/'results'/summary['solver_run']; training_run=Path(summary['initial_halting_measurements_preserved_at'])
    destination=runtime_run/'final_figures'; destination.mkdir(exist_ok=False)
    source_dir=runtime_run/'final_figures_source'; source_dir.mkdir(exist_ok=False)
    shutil.copyfile(__file__,source_dir/'finalize_artifacts.py')
    def export(fig,name):
        fig.savefig(destination/name,dpi=150,bbox_inches='tight'); plt.close(fig)
        shutil.copyfile(destination/name,BASE/'figures'/name)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for seed in (42,123,2026):
        d=dict(np.load(solver_run/f'B_seed{seed}_latent_diagnostics.npz'))
        axes[0].plot(range(1,17),d['change'].mean(0),label=f'seed {seed}')
        axes[1].plot(range(1,17),d['history_attention'].mean(0),label=f'history {seed}')
        axes[1].plot(range(1,17),d['input_attention'].mean(0),'--',label=f'input {seed}')
    axes[0].set(title='Latent RMS update (diagnostic only)',xlabel='Iteration',ylabel='RMS state change')
    axes[1].set(title='Input/history attention allocation',xlabel='Iteration',ylabel='Attention mass')
    for axis in axes: axis.legend(fontsize=7); axis.grid(alpha=.2)
    fig.tight_layout(); export(fig,'latent_attention_labeled.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True); training_diagnostics=[]
    for axis,seed_group in zip(axes,halts):
        seed=seed_group['selection']['seed']
        for r in seed_group['candidates']:
            tag=f'seed{seed}_cost{r["penalty"]:g}'; history=read(training_run/(tag+'_history.json'))
            epochs=[row['epoch'] for row in history]
            line,=axis.plot(epochs,[row['sampled_mean_steps'] for row in history],label=f'cost {r["penalty"]:g} sampled')
            axis.plot(epochs,[row['validation']['val']['steps'] for row in history],'--',color=line.get_color())
            for row in history:
                training_diagnostics.append({'seed':seed,'penalty':r['penalty'],'epoch':row['epoch'],
                    'sampled_mean_steps':row['sampled_mean_steps'],'sampled_success':row['sampled_success'],
                    'validation_mean_steps':row['validation']['val']['steps'],'validation_success':row['validation']['val']['success'],
                    'reinforce_surrogate_loss':row['reinforce_loss'],'mean_cost':row['cost'],
                    'derived_terminal_policy_nll_mean':row['cost']-2*(1-row['sampled_success'])-r['penalty']*row['sampled_mean_steps']/16})
        axis.set(title=f'Frozen solver | seed {seed}',xlabel='Halt-training epoch',ylim=(1,16.5)); axis.legend(fontsize=7); axis.grid(alpha=.2)
    axes[0].set_ylabel('Mean iterations (dashed = deterministic validation)'); fig.tight_layout(); export(fig,'halt_training.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for axis,s in zip(axes,halts):
        r=next(r for r in s['candidates'] if r['penalty']==0); row=r['groups']['seen']
        hist=np.asarray(row['step_histogram']); forced=np.zeros(16,dtype=int); forced[-1]=round(row['forced']*row['n'])
        autonomous=hist-forced; assert (autonomous>=0).all() and int(hist.sum())==row['n']
        axis.bar(range(1,17),autonomous,label='Autonomous halt',color='seagreen')
        axis.bar(range(1,17),forced,bottom=autonomous,label='Cap-forced stop',color='darkorange')
        axis.set(title=f'seed {r["seed"]} | cost 0 | seen',xlabel='Actual terminal iteration',xticks=[1,4,8,12,16]); axis.legend(fontsize=8)
    axes[0].set_ylabel('Number of official problems'); fig.tight_layout(); export(fig,'autonomous_vs_forced.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for axis,k in zip(axes,range(3)):
        selected=[next(r for r in s['candidates'] if r['penalty']==0) for s in halts]
        fixed=[r['representative_paired_latency'][k]['fixed16']['mean_ms'] for r in selected]
        adaptive=[r['representative_paired_latency'][k]['adaptive']['mean_ms'] for r in selected]
        x=np.arange(3); axis.bar(x-.18,fixed,.36,label='Fixed16'); axis.bar(x+.18,adaptive,.36,label='Halt policy')
        axis.set(title=f'Batch {selected[0]["latency"][k]["batch"]} | cost 0',xticks=x,xticklabels=['42','123','2026'],xlabel='Seed',ylabel='Model milliseconds'); axis.legend(fontsize=8)
    fig.suptitle('Paired varied-validation timing; zero-cost policy shown, not adopted'); fig.tight_layout(); export(fig,'representative_latency.png')
    data=dict(np.load(DATA/'dataset.npz')); records=read(DATA/'accepted_records.json')
    assert np.array_equal(data['field_keys'][data['field_id']],np.column_stack((data['map_id'],data['goal'])))
    paths={m:{r['record_index']:r for r in read(solver_run/f'{m}_seed42_paths.json')} for m in ('A','B','C')}
    ids=np.load(solver_run/'used_problem_ids.npz')['evaluation']; extra_cases=[]
    for m in (1,2,3,4):
        available=ids[data['map_id'][ids]==m]
        successes=[int(i) for i in available if paths['B'][int(i)]['status']=='success']
        i=successes[0] if successes else int(available[0]); r=records[i]
        w=data['walls'][m]; s=int(data['start'][i]); g=int(data['goal'][i]); field=data['fields'][data['field_id'][i]]
        oracle=decode(directions(w,field).astype(float),w,s,g); assert oracle['status']=='success' and len(oracle['path'])-1==data['optimal'][i]
        fig,axes=plt.subplots(1,4,figsize=(16,4.8))
        for axis,(label,route) in zip(axes,[('Oracle',oracle)]+[(mode,paths[mode][i]) for mode in ('A','B','C')]):
            axis.imshow(w.reshape(32,32),cmap='gray_r',vmin=0,vmax=1,origin='upper',interpolation='nearest')
            start_artist=axis.scatter(s%32,s//32,c='deepskyblue',s=25,edgecolors='black',zorder=4)
            goal_artist=axis.scatter(g%32,g//32,c='gold',s=60,marker='*',edgecolors='black',zorder=5)
            assert np.array_equal(start_artist.get_offsets()[0],[r['sx'],r['sy']]) and np.array_equal(goal_artist.get_offsets()[0],[r['gx'],r['gy']])
            p=np.array(route['path']); line,=axis.plot(p%32,p//32,color='seagreen' if route['status']=='success' else 'crimson',linewidth=1.5)
            assert np.array_equal(line.get_xdata(),p%32) and np.array_equal(line.get_ydata(),p//32)
            axis.set_title(f'{label}: {route["status"]}\nL={len(p)-1}, T={route.get("steps",0)}',fontsize=10)
        fig.suptitle(f'{r["map_name"]} | {r["scenario_id"]}\nS=({r["sx"]},{r["sy"]}) G=({r["gx"]},{r["gy"]}) | optimum={data["optimal"][i]} | seed42 | no repair',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.88)); name=f'extra_map{m}_case{i}.png'; export(fig,name)
        extra_cases.append({'map':MAPS[m],'record_index':i,'S':[r['sx'],r['sy']],'G':[r['gx'],r['gy']],
                            'optimal':int(data['optimal'][i]),'wall_sha256':hashlib.sha256(w.tobytes()).hexdigest(),'image':name,
                            'selection':'first B-success if one exists for this map, otherwise first problem; descriptive example only'})
    def group(mode,split): return next(r for r in summary['aggregate'] if r['mode']==mode and r['group']==split)
    a,b,c=[group(m,'seen')['success_mean']*100 for m in ('A','B','C')]
    b_unseen=group('B','unseen')['success_mean']*100
    by={(r['seed'],r['mode']):r for r in solvers}; seeds=(42,123,2026)
    wins_a=sum(by[(s,'B')]['groups']['seen']['success']>by[(s,'A')]['groups']['seen']['success'] for s in seeds)
    wins_c=sum(by[(s,'B')]['groups']['seen']['success']>by[(s,'C')]['groups']['seen']['success'] for s in seeds)
    adopted=sum(s['selection']['adopted_penalty'] is not None for s in halts)
    findings=['## 연구 질문에 대한 답','',
        '1. **잠재 상태도 학습되는가?** 학습 신호와 일부 실제 완주를 확인했습니다. B는 같은 공식16문제에서7,000스텝 후15/16완주·loss0.0787로 엄격 기준을 통과했습니다. A는2만스텝 후8/16완주로 미달했습니다. 잠재 상태가 특정 사고 내용을 나타낸다고 해석하지 않습니다.',
        f'2. **독립1회보다16회가 좋은가?** 관찰 지도 평균은 A={a:.2f}%, B={b:.2f}%로 B−A={b-a:+.2f}%p입니다. 그러나 B가 A보다 높은 시드는{wins_a}/3개였고 B의 표본SD는{group("B","seen")["success_std"]*100:.2f}%p였습니다. 평균 이득을 일관된 우위로 일반화할 수 없습니다.',
        f'3. **과거 기록이 도움이 되는가?** C={c:.2f}%이며 B−C={b-c:+.2f}%p, B가 높은 시드는{wins_c}/3개입니다. 정상 학습 B/C가 직접 비교이고, 기록 교란은 분포 밖 입력 진단입니다. Attention 비중이나 일부 교란 저하만으로 기록의 필수성을 입증하지 않습니다.',
        '4. **종료 시점을 학습했는가?** 검증에서는 모든 후보가16회까지 계속했습니다. 관찰 테스트도 평균16회·강제종료100%였습니다. 종료 헤드의 gradient·샘플링·로그확률은 검증됐지만, 유용한 조기 종료를 학습한 결과는 아닙니다.',
        f'5. **품질을 유지하면서 실제 계산을 줄였는가?** 채택된 자율 종료 정책은{adopted}/3개입니다. 이번 설정에서는 계산 절약에 실패해 고정16회를 유지합니다. 반복 수가 같으면 종료 판단의 추가 비용도 생깁니다.',
        f'6. **새 공식 지도에서도 풀리는가?** B의 미관찰 지도 평균 완주율은{b_unseen:.3f}%입니다. 일부 완주 사례는 있지만 실용적인 일반화 성능을 확보하지 못했습니다. 보유 지도는5개, 미관찰 평가는2개입니다.',
        '7. **다음에는 무엇 하나만 바꿀 것인가?** 구조·데이터·라벨·배치·학습률은 고정하고, 새 run에서 solver epoch 상한만80→160으로 늘려 최적화 부족의 영향을 먼저 분리하는 것이 적절합니다. 이번에는 후반 손실 개선과 시드 변동이 컸습니다. 공동 종료 미세조정은 학습 조건이 달라지므로 수행하지 않았고 별도 후속 실험으로 남깁니다.','',
        'A는 이번 실험의 **1회 Attention 모델**이며, 이전 SH001의 단일 MLP와 다른 기준선입니다. 또한 EXP004와는 감독 목표·정책 디코더가 달라 성공률 차이를 잠재 구조만의 효과로 해석할 수 없습니다.','']
    report_path=BASE/'REPORT.md'; body=report_path.read_text(encoding='utf-8')
    heading,rest=body.split('\n',1)
    body=heading+'\n\n'+'\n'.join(findings)+rest
    body=body.replace('figures/latent_attention.png','figures/latent_attention_labeled.png')
    additional=['','## 최종 추론 구현·지연 재검토','',
        f'종료 학습 run은 `{training_run.name}`이며 최종 추론 감사 run은 `{runtime_run.name}`입니다. 종료 헤드 로짓을 한 반복에서 중복 계산하던 부분을 한 번 계산해 재사용하도록 정리했습니다. 어떤 가중치도 재학습하지 않았으며, 모든9개 종료 후보의 전체 테스트 로짓 SHA-256·경로·지표와 검증 결과가 원래 구현과 정확히 같았습니다. 원래 측정과 보고서도 감사 run의 initial_* 및 원래 학습 run에 보존했습니다.',
        '초기 지연 측정은 첫 문제/배치 반복이었습니다. 최종 측정은 훈련 기준Q1~Q4로 층화한 공식 검증 문제를 섞어 batch1/16/64를 짝지어 측정하고 순서를 번갈았습니다. 모델만의GPU-resident지연이며 디코더·입출력전송은 제외합니다. 최대값은 측정 표본에서 관측된 값이며 모든 문제의 최악 지연 보장은 아닙니다. 테스트 성능으로 정책을 선택하지 않았습니다.','',
        '| seed | lambda | 검증 평균T | batch1 고정16 ms | 종료정책 ms | 종료정책 관측max ms | 채택 |','|---|---:|---:|---:|---:|---:|---|']
    for s in halts:
        for r in s['candidates']:
            t=r['representative_paired_latency'][0]
            additional.append(f'| {r["seed"]} | {r["penalty"]} | {r["validation"]["val"]["steps"]:.2f} | {t["fixed16"]["mean_ms"]:.3f} ± {t["fixed16"]["std_ms"]:.3f} | {t["adaptive"]["mean_ms"]:.3f} ± {t["adaptive"]["std_ms"]:.3f} | {t["adaptive"]["max_ms"]:.3f} | {r["adoptable"]} |')
    additional+=['','![다양한 검증 문제에서 실제 지연](figures/representative_latency.png)','',
        '이 표의 ±는 측정 시행 간 표준편차입니다. 전체batch크기별 평균·표준편차·관측최대·대상문제ID는 SUMMARY.json에 있습니다. 평균 FLOPs는 각 실제 종료 분포의 F(T)를 평균하며, 고정 인코더·최종 출력 비용과 종료 헤드까지 포함합니다. 단순히 평균T/16을 전체 연산 비율로 사용하지 않습니다.','',
        '![학습 중 종료 횟수 변화](figures/halt_training.png)','',
        '실선은 확률적으로 샘플링한 훈련 평균T, 점선은 결정적 검증 평균T입니다. REINFORCE surrogate 값은 최종 정책 NLL과 다른 값입니다. epoch별 평균 terminal NLL은 기록된 평균cost−2×실패율−lambda×평균T/16으로도 복원해 SUMMARY.json에 넣었습니다.','',
        '![자율 종료와 상한 강제 종료 구분](figures/autonomous_vs_forced.png)','',
        '위 종료 분포 그림은 각seed의0비용 정책입니다. 다른 비용 후보도 검증·관찰 테스트에서는 모두16회 강제종료였습니다. 이 결과는 동결16회 solver와 시험한 비용0/0.001/0.01에서의 결과이며, 다른 비용·공동 학습·다른 구조에서도 종료가 불가능하다는 뜻은 아닙니다.','',
        '## 지도별 실제 경로 추가 사례','',
        '아래는seed42의 같은 공식 문제에 대한 Oracle/A/B/C입니다. 각 지도에서 B 완주가 있으면 첫 완주 사례, 없으면 첫 문제를 선택했습니다. 선택된 사례로 전체 성능을 추정하지 않습니다.','']
    for r in extra_cases: additional.append(f'![{r["map"]} 실제 경로](figures/{r["image"]})\n')
    additional+=['','## 최종 감사 범위와 남은 한계','',
        '전체 방향 정책은 같은 지도·G에서는 S와 무관한 정답입니다. S는 실제 입력에 들어가지만 이 손실이 S 의존 처리를 강제하지는 않습니다. 같은 지도·G가 분할을 넘지 않도록 기존 정책을 유지했습니다. 원시 방향 argmax의 벽 충돌·재방문이 주요 실패였고, 정답 탐색이나 합법 방향 마스킹으로 복구하지 않았습니다.',
        '기존 H001/SH001 및 공용data의3,799파일은 실행 전후 SHA-256이 같았습니다. 학습 run의 소스·설정·optimizer·RNG와 체크포인트는 보존했습니다. 데이터 전용 Phase0의 초기 소스 스냅샷 누락은 가장 이른 후속 불변 스냅샷의 동일 모듈 버전에서 복구해 recovered_source_provenance.json에 출처와 늦게 보관한 사실을 명시했습니다. 데이터 전용 단계에는 학습optimizer/RNG가 적용되지 않습니다.',
        '통계는3seed의 표본SD이며 유의성이나 모집단 비열등성을 증명하지 않습니다. 문제가 같은 지도·목표를 공유해 독립이 아니고 지도 수가 작습니다. 가중치 공유·Attention 비중·잠재 변화량만으로 사고의 내용을 해석하지 않습니다. 종료 실패와 solver 학습 부족의 가능성도 구분해야 합니다.',
        '실행 명령·재개 조건·run별 용도는 README.md와 RUNS.json에 있습니다. 원본/문제별 대용량 결과는 기존ignore정책으로 로컬에, 요약·보고서·대표그림은 Git 관리 대상으로 보존합니다.','']
    report_path.write_text(body+'\n'.join(additional),encoding='utf-8')
    summary['halt_training_diagnostics']=training_diagnostics; summary['extra_case_visualizations']=extra_cases
    summary['release_source_sha256']={**source_hashes(),**{f'tools/{p.name}':sha(p) for p in (BASE/'tools').glob('*.py')}}
    summary['extra_figure_source_sha256']=sha(__file__); summary['field_map_goal_indices_exact']=True
    write(BASE/'SUMMARY.json',summary)
    extra_provenance={'runtime_run':runtime_run.name,'source_sha256':sha(__file__),'cases':extra_cases,
                      'files':{p.name:sha(p) for p in destination.glob('*.png')},'official_coordinates_and_drawn_lines_verified':True}
    write(destination/'provenance.json',extra_provenance); write(BASE/'figures/extra_provenance.json',extra_provenance)
    root=ROOT/'README.md'; text=root.read_text(encoding='utf-8').replace('잠재 Attention·최종 방향 정책·자율 종료 구현 및 실험 진행','3seed 잠재 Attention·종료 정책 실험 완료, 반복 이득 변동·성능 유지형 계산 절약 실패')
    root.write_text(text,encoding='utf-8')
    print('FINAL_ARTIFACTS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('runtime_run',type=Path); args=parser.parse_args(); finalize(args.runtime_run)
