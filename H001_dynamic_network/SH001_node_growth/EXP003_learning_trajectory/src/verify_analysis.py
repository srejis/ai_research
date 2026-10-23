"""Read-only integrity and independent endpoint/aggregation audit. No CUDA needed."""
import argparse
import csv
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

from core import ROOT, PROJECT, json_read, json_write, digest


def rows(path):
    with Path(path).open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def close(value,expected):
    assert np.isclose(float(value),expected,rtol=1e-10,atol=1e-12),(value,expected)


def verify(run_id):
    root=ROOT/"results"/run_id
    analysis=root/"analysis"
    status=json_read(analysis/"summary.json")
    assert status["status"]=="COMPLETE"
    count=0
    source_versions=[]
    for seed in status["seeds"]:
        directory=root/f"seed_{seed}"
        assert json_read(directory/"verification.json")["success"]
        for relative,expected in json_read(directory/"artifact_sha256.json").items():
            assert digest(ROOT/relative)==expected,relative
            count+=1
        for relative,expected in json_read(directory/"protected_previous_experiments.json").items():
            assert digest(PROJECT/relative)==expected,relative
        versions=json_read(directory/"environment.json")["code_sha256"]
        for name,value in versions.items():
            assert digest(directory/"source"/name)==value
            assert digest(ROOT/"src"/name)==value
        source_versions.append(versions)
    assert all(value==source_versions[0] for value in source_versions)
    for relative,expected in json_read(analysis/"artifact_sha256.json").items():
        assert digest(ROOT/relative)==expected,relative
        count+=1
    paired=rows(analysis/"paired_seed_metrics.csv")
    for row in paired:
        seed,condition=int(row["seed"]),row["condition"]
        data=rows(root/f"seed_{seed}/branches/{condition}/performance.csv")
        control=sorted((r for r in data if r["branch"]=="control"),key=lambda r:int(r["additional_step"]))
        growth=sorted((r for r in data if r["branch"]=="growth"),key=lambda r:int(r["additional_step"]))
        assert len(control)==len(growth)==int(row["observation_steps"])+1
        close(row["accuracy_difference_pp"],(float(growth[-1]["validation_accuracy"])-float(control[-1]["validation_accuracy"]))*100)
        close(row["loss_difference"],float(growth[-1]["validation_loss"])-float(control[-1]["validation_loss"]))
        for field,prefix in (("cumulative_train_wall_seconds","equal_time"),("forward_mac_proxy","equal_mac")):
            budget=min(float(control[-1][field]),float(growth[-1][field]))
            c=next(r for r in reversed(control) if float(r[field])<=budget)
            g=next(r for r in reversed(growth) if float(r[field])<=budget)
            close(row[prefix+"_control_steps"],int(c["additional_step"]))
            close(row[prefix+"_growth_steps"],int(g["additional_step"]))
            close(row[prefix+"_loss_difference"],float(g["validation_loss"])-float(c["validation_loss"]))
    for row in rows(analysis/"paired_aggregate.csv"):
        values=np.array([float(r[row["metric"]]) if r[row["metric"]] else np.nan for r in paired if r["condition"]==row["condition"]])
        values=values[np.isfinite(values)]
        assert len(values)==int(row["n"])
        if len(values): close(row["mean"],values.mean())
        if len(values)>1: close(row["sample_std"],values.std(ddof=1))
    for row in rows(analysis/"threshold_steps.csv"):
        p=root/f"seed_{int(row['seed'])}/branches/{row['condition']}/performance.csv"
        data=sorted([r for r in rows(p) if r["branch"]==row["branch"]],key=lambda r:int(r["additional_step"]))
        first=next((r for r in data if float(r["validation_accuracy"])>=float(row["accuracy_threshold"])),None)
        assert (first is not None)==(row["reached"]=="True")
        if first is not None: close(row["additional_steps"],int(first["additional_step"]))
        else: assert row["additional_steps"]==""
    result={"success":True,"checked_artifact_files":count,"seeds":status["seeds"],
        "paired_runs":len(paired),"same_training_source_hashes_across_seeds":True,
        "independent_endpoint_budget_threshold_and_sample_std_checks":True,
        "EXP001_EXP002_all_files_unchanged":True,"training_or_validation_inference_performed":False}
    output=root/"final_verification.json"
    if output.exists(): assert json_read(output)==result
    else: json_write(output,result)
    print(f"FINAL VERIFIED: {len(paired)} pairs, {count} artifact hashes, endpoints/budgets/thresholds/sample SD, previous experiments unchanged.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True)
    verify(parser.parse_args().run_id)
