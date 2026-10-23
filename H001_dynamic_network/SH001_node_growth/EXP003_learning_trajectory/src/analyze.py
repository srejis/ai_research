"""Within-seed paired metrics, cross-seed summaries, process plots, Q1-Q5 report."""
import argparse
import csv
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import ROOT, json_read, json_write, digest

CONDITIONS=("P25","P50","P75","P100")
LAYERS=("fc1","fc2","fc3")


def read(path):
    with Path(path).open(encoding="utf-8",newline="") as stream:
        result=list(csv.DictReader(stream))
    for row in result:
        for key,value in row.items():
            if key in {"branch","layer","logits_sha256"}: continue
            row[key]=float(value) if value!="" else np.nan
    return result


def write_csv(path,rows):
    with Path(path).open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def stats(values):
    values=np.array(values,dtype=float)
    values=values[np.isfinite(values)]
    return (float(values.mean()) if len(values) else None,
            float(values.std(ddof=1)) if len(values)>1 else None,len(values))


def fmt(value,digits=5):
    return "미달/결측" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def moving(values,window=25):
    a=np.asarray(values,float)
    if len(a)<window: return a
    # Missing values remain missing rather than becoming direction=0.
    valid=np.isfinite(a).astype(float)
    numerator=np.convolve(np.nan_to_num(a),np.ones(window),mode="same")
    denominator=np.convolve(valid,np.ones(window),mode="same")
    return np.divide(numerator,denominator,out=np.full_like(a,np.nan),where=denominator>0)


def analyze(run_id,seeds):
    root=ROOT/"results"/run_id
    output=root/"analysis"
    output.mkdir(exist_ok=False)
    figures=output/"figures"; figures.mkdir()
    cfg=json_read(ROOT/"configs/experiment.json")
    summaries,performance,directions,new={}, {}, {}, {}
    pair_metrics,layer_metrics,thresholds=[],[],[]
    for seed in seeds:
        directory=root/f"seed_{seed}"
        assert json_read(directory/"verification.json")["success"]
        summaries[seed]=json_read(directory/"summary.json")
        for condition in CONDITIONS:
            pair_dir=directory/"branches"/condition
            perf=read(pair_dir/"performance.csv")
            control=sorted([r for r in perf if r["branch"]=="control"],key=lambda r:r["additional_step"])
            growth=sorted([r for r in perf if r["branch"]=="growth"],key=lambda r:r["additional_step"])
            performance[seed,condition]={"control":control,"growth":growth}
            direction=read(pair_dir/"directions.csv")
            directions[seed,condition]={layer:[r for r in direction if r["layer"]==layer] for layer in (*LAYERS,"all")}
            new[seed,condition]=read(pair_dir/"new_neuron.csv")
            c,g=control[-1],growth[-1]
            equal_time=min(c["cumulative_train_wall_seconds"],g["cumulative_train_wall_seconds"])
            matched_c=max((r for r in control if r["cumulative_train_wall_seconds"]<=equal_time),key=lambda r:r["additional_step"])
            matched_g=max((r for r in growth if r["cumulative_train_wall_seconds"]<=equal_time),key=lambda r:r["additional_step"])
            equal_macs=min(c["forward_mac_proxy"],g["forward_mac_proxy"])
            mac_c=max((r for r in control if r["forward_mac_proxy"]<=equal_macs),key=lambda r:r["additional_step"])
            mac_g=max((r for r in growth if r["forward_mac_proxy"]<=equal_macs),key=lambda r:r["additional_step"])
            all_dir=directions[seed,condition]["all"]
            divergence=np.array([1-r["update_cosine"] for r in all_dir])
            loss_difference=np.array([gr["validation_loss"]-cr["validation_loss"] for cr,gr in zip(control[1:],growth[1:])])
            correlation=float(np.corrcoef(divergence,loss_difference)[0,1]) if np.std(divergence)>0 and np.std(loss_difference)>0 else None
            pair_metrics.append({"seed":seed,"condition":condition,"source_step":summaries[seed]["branch_steps"][condition],
                "observation_steps":summaries[seed]["S_complete"],"control_accuracy":c["validation_accuracy"],"growth_accuracy":g["validation_accuracy"],
                "accuracy_difference_pp":(g["validation_accuracy"]-c["validation_accuracy"])*100,
                "control_loss":c["validation_loss"],"growth_loss":g["validation_loss"],"loss_difference":g["validation_loss"]-c["validation_loss"],
                "control_train_wall_seconds":c["cumulative_train_wall_seconds"],"growth_train_wall_seconds":g["cumulative_train_wall_seconds"],
                "wall_ratio_growth_control":g["cumulative_train_wall_seconds"]/c["cumulative_train_wall_seconds"],
                "control_gpu_seconds":c["cumulative_gpu_seconds"],"growth_gpu_seconds":g["cumulative_gpu_seconds"],
                "samples_each":c["samples"],"control_forward_mac_proxy":c["forward_mac_proxy"],"growth_forward_mac_proxy":g["forward_mac_proxy"],
                "equal_time_budget_seconds":equal_time,"equal_time_control_steps":matched_c["additional_step"],"equal_time_growth_steps":matched_g["additional_step"],
                "equal_time_loss_difference":matched_g["validation_loss"]-matched_c["validation_loss"],
                "equal_time_accuracy_difference_pp":(matched_g["validation_accuracy"]-matched_c["validation_accuracy"])*100,
                "equal_mac_control_steps":mac_c["additional_step"],"equal_mac_growth_steps":mac_g["additional_step"],
                "equal_mac_loss_difference":mac_g["validation_loss"]-mac_c["validation_loss"],
                "loss_reduction_per_100_steps_difference":((growth[0]["validation_loss"]-g["validation_loss"])-(control[0]["validation_loss"]-c["validation_loss"]))*100/summaries[seed]["S_complete"],
                "direction_vs_loss_gap_pearson_descriptive":correlation})
            for label,rows in (("control",control),("growth",growth)):
                for target in cfg["accuracy_thresholds"]:
                    first=next((r for r in rows if r["validation_accuracy"]>=target),None)
                    thresholds.append({"seed":seed,"condition":condition,"branch":label,"accuracy_threshold":target,
                        "reached":first is not None,"additional_steps":first["additional_step"] if first else None,
                        "train_wall_seconds":first["cumulative_train_wall_seconds"] if first else None})
            for layer,rows in directions[seed,condition].items():
                first=next((r["additional_step"] for r in rows if np.isfinite(r["update_cosine"]) and r["update_cosine"]<cfg["direction_change_cosine"]),None)
                first_gradient=next((r["additional_step"] for r in rows if np.isfinite(r["gradient_cosine"]) and r["gradient_cosine"]<cfg["direction_change_cosine"]),None)
                update=np.array([r["update_cosine"] for r in rows]); grad=np.array([r["gradient_cosine"] for r in rows])
                quarter=max(1,len(rows)//4)
                layer_metrics.append({"seed":seed,"condition":condition,"layer":layer,"first_update_cosine_below_099":first,
                    "first_gradient_cosine_below_099":first_gradient,"mean_update_cosine":np.nanmean(update),"mean_gradient_cosine":np.nanmean(grad),
                    "early_quarter_update_cosine":np.nanmean(update[:quarter]),"late_quarter_update_cosine":np.nanmean(update[-quarter:]),
                    "fraction_updates_below_099":float(np.mean(update<cfg["direction_change_cosine"])),
                    "minimum_update_cosine":np.nanmin(update),"final_cumulative_cosine":rows[-1]["cumulative_cosine"],
                    "final_path_distance":rows[-1]["path_distance"],"final_relative_path_distance":rows[-1]["relative_path_distance"]})
    write_csv(output/"paired_seed_metrics.csv",pair_metrics)
    write_csv(output/"direction_seed_metrics.csv",layer_metrics)
    write_csv(output/"threshold_steps.csv",thresholds)
    aggregate=[]
    fields=[k for k in pair_metrics[0] if k not in {"seed","condition","source_step","observation_steps"}]
    for condition in CONDITIONS:
        for metric in fields:
            mean,std,n=stats([r[metric] for r in pair_metrics if r["condition"]==condition])
            aggregate.append({"condition":condition,"metric":metric,"mean":mean,"sample_std":std,"n":n})
    write_csv(output/"paired_aggregate.csv",aggregate)
    threshold_pairs=[]
    for condition in CONDITIONS:
        for target in cfg["accuracy_thresholds"]:
            differences=[]; achieved={"control":0,"growth":0}
            for seed in seeds:
                items={r["branch"]:r for r in thresholds if r["seed"]==seed and r["condition"]==condition and r["accuracy_threshold"]==target}
                for label in achieved: achieved[label]+=items[label]["reached"]
                if items["control"]["reached"] and items["growth"]["reached"]:
                    differences.append(items["growth"]["additional_steps"]-items["control"]["additional_steps"])
            mean,std,n=stats(differences)
            threshold_pairs.append({"condition":condition,"target":target,"control_reached_seeds":achieved["control"],"growth_reached_seeds":achieved["growth"],
                "paired_achieved_seeds":n,"growth_minus_control_steps_mean":mean,"sample_std":std})
    write_csv(output/"threshold_aggregate.csv",threshold_pairs)
    grid=np.linspace(0,1,201)
    aligned=[]

    def save(fig,name):
        fig.tight_layout(); fig.savefig(figures/name,dpi=150); plt.close(fig)

    def mean_curve(ax,xy,label,color,record=None):
        values=np.stack([np.interp(grid,x,y) for x,y in xy])
        mu=np.nanmean(values,axis=0); sd=np.nanstd(values,axis=0,ddof=1)
        ax.plot(grid,mu,label=label,color=color)
        ax.fill_between(grid,mu-sd,mu+sd,color=color,alpha=0.14)
        for x,y in xy: ax.plot(x,y,color=color,alpha=0.12,linewidth=0.6)
        if record:
            for t,m,s in zip(grid,mu,sd): aligned.append({"series":record,"normalized_additional_step":t,"mean":m,"sample_std":s})

    for seed in seeds:
        fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True)
        for i,condition in enumerate(CONDITIONS):
            for j,layer in enumerate(LAYERS):
                ax=axes[i,j]; rows=directions[seed,condition][layer]; x=[r["additional_step"] for r in rows]
                for key,label,color in (("update_cosine","Actual update","tab:blue"),("gradient_cosine","Task gradient","tab:orange")):
                    vals=[r[key] for r in rows]
                    ax.plot(x,vals,color=color,alpha=0.12,linewidth=0.4)
                    ax.plot(x,moving(vals),color=color,label=label)
                ax.axhline(0.99,color="gray",linestyle=":",linewidth=0.8)
                ax.set(title=f"{condition} (source {summaries[seed]['branch_steps'][condition]}): {layer}",ylabel="Cosine",xlabel="Additional optimizer step",ylim=(-1.02,1.02))
                if i==0 and j==0: ax.legend(fontsize=8)
        fig.suptitle(f"1. Seed {seed}: stepwise directions (raw faint; centered 25-step mean)")
        save(fig,f"01_directions_seed_{seed}.png")
    for key,label,name in (("validation_loss","Validation loss","02_validation_loss.png"),("validation_accuracy","Validation accuracy","03_validation_accuracy.png")):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,condition in zip(axes.flat,CONDITIONS):
            for branch,color in (("control","tab:blue"),("growth","tab:orange")):
                xy=[]
                for seed in seeds:
                    rows=performance[seed,condition][branch]
                    xy.append((np.array([r["additional_step"] for r in rows])/summaries[seed]["S_complete"],np.array([r[key] for r in rows])))
                mean_curve(ax,xy,branch,color,f"{condition}.{key}.{branch}")
            ax.set(title=condition,xlabel="Additional step / seed-specific S_complete",ylabel=label)
            ax.grid(alpha=0.2); ax.legend()
        fig.suptitle("Mean ± sample SD across 3 seeds; individual paths faint")
        save(fig,name)
    for key,title,name in (("relative_path_distance","4. Normalized original-weight path separation","04_path_distance.png"),
                            ("cumulative_cosine","7. Cumulative update direction","07_cumulative_direction.png")):
        fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True)
        for i,condition in enumerate(CONDITIONS):
            for j,layer in enumerate(LAYERS):
                ax=axes[i,j]
                xy=[(np.array([r["additional_step"] for r in directions[seed,condition][layer]])/summaries[seed]["S_complete"],np.array([r[key] for r in directions[seed,condition][layer]])) for seed in seeds]
                mean_curve(ax,xy,layer,"tab:purple",f"{condition}.{layer}.{key}")
                ax.set(title=f"{condition}: {layer}",xlabel="Additional step / S_complete",ylabel=key)
                ax.grid(alpha=0.2)
        fig.suptitle(title+" (mean ± sample SD)")
        save(fig,name)
    for fields,title,name in [
        (("input_weight_norm","output_weight_norm","input_gradient_norm","output_gradient_norm"),"5a. New neuron weights and task gradients","05a_new_neuron_weights_gradients.png"),
        (("relu_active_fraction","contribution_relative_norm"),"5b. New neuron activation and pre-activation contribution","05b_new_neuron_activation.png")]:
        fig,axes=plt.subplots(4,len(fields),figsize=(4*len(fields),12),squeeze=False,sharex=True)
        for i,condition in enumerate(CONDITIONS):
            for j,field in enumerate(fields):
                ax=axes[i,j]
                xy=[(np.array([r["additional_step"] for r in new[seed,condition]])/summaries[seed]["S_complete"],np.array([r[field] for r in new[seed,condition]])) for seed in seeds]
                mean_curve(ax,xy,field,"tab:green",f"{condition}.new.{field}")
                ax.set(title=f"{condition}: {field}",xlabel="Additional step / S_complete")
                ax.grid(alpha=0.2)
        fig.suptitle(title+" (mean ± sample SD)")
        save(fig,name)
    fig,axes=plt.subplots(4,3,figsize=(16,12))
    for i,condition in enumerate(CONDITIONS):
        for seed,color in zip(seeds,("tab:blue","tab:orange","tab:green")):
            for branch,style in (("control","-"),("growth","--")):
                rows=performance[seed,condition][branch]
                for ax,key,scale in ((axes[i,0],"additional_step",1),(axes[i,1],"cumulative_train_wall_seconds",1),(axes[i,2],"forward_mac_proxy",1e9)):
                    ax.plot(np.array([r[key] for r in rows])/scale,[r["validation_loss"] for r in rows],style,color=color,label=f"{seed} {branch}",linewidth=0.8)
        for ax,label in zip(axes[i],("Additional steps","Measured synchronized training wall seconds","Forward dense-MAC proxy (billions)")):
            ax.set(title=condition,xlabel=label,ylabel="Validation loss"); ax.grid(alpha=0.2)
        if i==0: axes[i,0].legend(fontsize=6)
    fig.suptitle("6. Step efficiency vs measured time vs forward-work proxy (individual seeds)")
    save(fig,"06_efficiency.png")
    fig,ax=plt.subplots(figsize=(10,5))
    for seed,color in zip(seeds,("tab:blue","tab:orange","tab:green")):
        directory=root/f"seed_{seed}"
        exact=read(directory/"baseline/validation_exact.csv")
        coarse=read(directory/"baseline/validation_coarse.csv")
        ax.plot([r["step"] for r in exact],[r["accuracy"] for r in exact],color=color,linewidth=0.7,label=f"seed {seed}, first={summaries[seed]['S_complete']}")
        ax.scatter([r["step"] for r in coarse],[r["accuracy"] for r in coarse],color=color,s=7,alpha=0.5)
    ax.axhline(0.95,color="black",linestyle="--"); ax.set(xlabel="Baseline optimizer step",ylabel="Validation accuracy",title="8. Exact first crossing vs coarse 25-step observations")
    ax.legend(); ax.grid(alpha=0.2); save(fig,"08_baseline_completion.png")
    write_csv(output/"aligned_curve_statistics.csv",aligned)

    from report import render
    render(output, root, seeds, summaries, pair_metrics, layer_metrics, aggregate, threshold_pairs, new)
    shutil.copy2(Path(__file__),output/"analyze_source.py")
    json_write(output/"summary.json",{"status":"COMPLETE","seeds":seeds,"S_complete":{str(s):summaries[s]["S_complete"] for s in seeds},
        "paired_runs":len(pair_metrics),"all_seed_verification_passed":True,"std_definition":"sample std ddof=1",
        "previous_experiments_unchanged":True})
    json_write(output/"artifact_sha256.json",{p.relative_to(ROOT).as_posix():digest(p) for p in output.rglob("*") if p.is_file()})
    print(f"ANALYSIS COMPLETE: {output}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True); parser.add_argument("--seeds",type=int,nargs="+",default=[42,123,2026])
    args=parser.parse_args(); analyze(args.run_id,args.seeds)
