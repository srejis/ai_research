"""Independent audit of persisted XOR paths, optimizer mapping and validation."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F

from core import ROOT, PROJECT, json_read, json_write, digest, load, setup, data, evaluate, flatten, finite
from model import TinyMLP


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def vector_from_checkpoint(cp):
    return np.concatenate([cp["model_state_dict"][name].numpy().ravel() for name in cp["layout"]])


def decode(directory):
    current = np.load(directory / "initial.npy").view(np.uint32).copy()
    yield 0, current.view(np.float32).copy()
    for path in sorted(directory.glob("steps_*.npz")):
        with np.load(path) as archive:
            for step, mask in zip(archive["steps"], archive["xor"]):
                current = np.bitwise_xor(current, mask)
                yield int(step), current.view(np.float32).copy()


def old_layers(vector, mapping):
    result = {}
    for layer in ("fc1", "fc2", "fc3"):
        result[layer] = np.concatenate([vector[s["start"]:s["end"]] for name, s in mapping.items() if name.startswith(layer+".")]).astype(np.float64)
    result["all"] = np.concatenate(list(result.values()))
    return result


def close(actual, expected):
    assert np.isclose(float(actual), expected, rtol=1e-9, atol=2e-12), (actual, expected)


def cs(a,b):
    norms = np.linalg.norm(a),np.linalg.norm(b)
    return None if min(norms)<=1e-12 else float(np.clip(np.dot(a,b)/(norms[0]*norms[1]),-1,1))


def verify(seed, run_id):
    directory = ROOT / "results" / run_id / f"seed_{seed}"
    checkpoints = ROOT / "checkpoints" / run_id / f"seed_{seed}"
    summary, cfg = json_read(directory/"summary.json"), json_read(directory/"config.json")
    for relative,expected in json_read(directory/"artifact_sha256.json").items():
        assert digest(ROOT/relative)==expected, relative
    for relative,expected in json_read(directory/"protected_previous_experiments.json").items():
        assert digest(PROJECT/relative)==expected, relative
    complete = summary["S_complete"]
    val_rows = rows(directory/"baseline/validation_exact.csv")
    assert [int(r["step"]) for r in val_rows] == list(range(complete+1))
    assert all(float(r["accuracy"])<0.95 for r in val_rows[:-1]) and float(val_rows[-1]["accuracy"])>=0.95
    completion = load(checkpoints/"baseline/completion.pt")
    assert completion["step"]==complete
    target_cps = {0:load(checkpoints/"baseline/step_00000.pt"),complete:completion}
    target_cps.update({v:load(checkpoints/k/"source.pt") for k,v in summary["branch_steps"].items()})
    baseline_steps = 0
    shared_control_hashes = {}
    for step, vector in decode(directory/"baseline/path"):
        if step in target_cps:
            assert np.array_equal(vector,vector_from_checkpoint(target_cps[step]))
        shared_control_hashes[step] = hashlib.sha256(vector.tobytes()).hexdigest()
        baseline_steps=step
    assert baseline_steps>=complete
    setup(seed)
    (_,_,vx,vy),_ = data(cfg,"cuda")
    validation_checks, direction_checks, checkpoint_checks = 0,0,0
    max_packed_error=0.0
    for condition,source_step in summary["branch_steps"].items():
        pair_dir=directory/"branches"/condition
        pair_cp=checkpoints/condition
        source=load(pair_cp/"source.pt")
        initial={label:load(pair_cp/label/"initial.pt") for label in ("control","growth")}
        assert initial["control"]["validation"] == initial["growth"]["validation"]
        for label in initial:
            cp=initial[label]
            assert torch.equal(cp["rng"]["cpu"],source["rng"]["cpu"])
            assert all(torch.equal(a,b) for a,b in zip(cp["rng"]["cuda"],source["rng"]["cuda"]))
            for name,value in source["model_state_dict"].items():
                assert torch.equal(value,cp["model_state_dict"][name])
            source_group=source["optimizer_state_dict"]["param_groups"][0]
            target_group=cp["optimizer_state_dict"]["param_groups"][0]
            assert source_group==target_group
            for key,state in source["optimizer_state_dict"]["state"].items():
                for field,value in state.items():
                    assert torch.equal(value,cp["optimizer_state_dict"]["state"][key][field])
            if label=="growth":
                new_ids=cp["optimizer_state_dict"]["param_groups"][1]["params"]
                assert all(i not in cp["optimizer_state_dict"]["state"] for i in new_ids)
            model=TinyMLP(cp["growth"]).cuda()
            model.load_state_dict(cp["model_state_dict"],strict=True)
            assert evaluate(model,vx,vy,cfg["validation_batch_size"])==cp["validation"]
            validation_checks+=1
        expected={label:{} for label in initial}
        for label in initial:
            for path in (pair_cp/label).glob("*.pt"):
                cp=load(path)
                assert finite(cp)
                assert all(p.device.type=="cpu" for p in cp["model_state_dict"].values())
                extra=cp["extra"]["additional_step"]
                expected[label][extra]=cp
                model=TinyMLP(cp["growth"]).cuda()
                model.load_state_dict(cp["model_state_dict"],strict=True)
                assert evaluate(model,vx,vy,cfg["validation_batch_size"])==cp["validation"]
                validation_checks+=1
                checkpoint_checks+=1
                if cp["growth"] and extra==complete:
                    state={n:t.cuda() for n,t in cp["model_state_dict"].items()}
                    w1=torch.cat([state["fc1.weight"],state["new_in.weight"]],0)
                    b1=torch.cat([state["fc1.bias"],state["new_in.bias"]],0)
                    w2=torch.cat([state["fc2.weight"],state["new_out"]],1)
                    with torch.no_grad():
                        packed=F.linear(F.relu(F.linear(F.relu(F.linear(vx,w1,b1)),w2,state["fc2.bias"])),state["fc3.weight"],state["fc3.bias"])
                        split=model(vx)
                        max_packed_error=max(max_packed_error,(packed-split).abs().max().item())
                        assert torch.allclose(packed,split,atol=1e-5,rtol=1e-5)
            final=expected[label][complete]
            assert torch.equal(final["rng"]["cpu"],source["rng"]["cpu"])
            assert all(torch.equal(a,b) for a,b in zip(final["rng"]["cuda"],source["rng"]["cuda"]))
            groups=final["optimizer_state_dict"]["param_groups"]
            for pid in groups[0]["params"]:
                assert final["optimizer_state_dict"]["state"][pid]["step"].item()==source_step+complete
            if label=="growth":
                for pid in groups[1]["params"]:
                    assert final["optimizer_state_dict"]["state"][pid]["step"].item()==complete
        scalar={(int(r["additional_step"]),r["layer"]):r for r in rows(pair_dir/"directions.csv")}
        c_initial=old_layers(vector_from_checkpoint(initial["control"]),initial["control"]["layout"])
        previous=None
        for (sc,vc),(sg,vg) in zip(decode(pair_dir/"control_path"),decode(pair_dir/"growth_path")):
            assert sc==sg
            for label,vec in (("control",vc),("growth",vg)):
                if sc in expected[label]:
                    assert np.array_equal(vec,vector_from_checkpoint(expected[label][sc]))
            c=old_layers(vc,initial["control"]["layout"])
            g=old_layers(vg,initial["growth"]["layout"])
            absolute = source_step + sc
            actual_hash = hashlib.sha256(vc.tobytes()).hexdigest()
            if absolute in shared_control_hashes:
                assert shared_control_hashes[absolute] == actual_hash, (condition, absolute)
            else:
                shared_control_hashes[absolute] = actual_hash
            if previous is not None:
                for layer in c:
                    uc,ug=c[layer]-previous[0][layer],g[layer]-previous[1][layer]
                    row=scalar[sc,layer]
                    expected_cos=cs(uc,ug)
                    if expected_cos is None: assert row["update_cosine"]==""
                    else: close(row["update_cosine"],expected_cos)
                    close(row["control_update_norm"],np.linalg.norm(uc))
                    close(row["growth_update_norm"],np.linalg.norm(ug))
                    close(row["path_distance"],np.linalg.norm(g[layer]-c[layer]))
                    close(row["relative_path_distance"],np.linalg.norm(g[layer]-c[layer])/(np.linalg.norm(c_initial[layer])+1e-12))
                    expected_cum=cs(c[layer]-c_initial[layer],g[layer]-c_initial[layer])
                    if expected_cum is None: assert row["cumulative_cosine"]==""
                    else: close(row["cumulative_cosine"],expected_cum)
                    if sc==1:
                        assert np.array_equal(uc,ug), (condition,layer)
                    direction_checks+=1
            previous=c,g
        assert sc==complete
        gc=load(pair_dir/"control_first_task_gradients.pt")
        gg=load(pair_dir/"growth_first_task_gradients.pt")
        for name in gc:
            assert torch.equal(gc[name],gg[name]),name
        assert torch.count_nonzero(gg["new_in.weight"])==0 and torch.count_nonzero(gg["new_in.bias"])==0
        new_rows=rows(pair_dir/"new_neuron.csv")
        for key,column in (("input","input_gradient_norm"),("output","output_gradient_norm")):
            nonzero=next((int(r["additional_step"]) for r in new_rows if float(r[column])>1e-12),None)
            assert nonzero==summary["pairs"][condition]["first_nonzero_new_gradient"][key]
        # Independently replay only the sampler, without any optimizer training.
        stream=source["batch_stream"]
        generator=torch.Generator(); generator.set_state(stream["generator"])
        cursor,order=stream["cursor"],stream["order"].clone()
        h=hashlib.sha256()
        for _ in range(complete):
            if cursor==55000: order=torch.randperm(55000,generator=generator); cursor=0
            indices=order[cursor:min(cursor+128,55000)]
            cursor+=len(indices); h.update(indices.numpy().tobytes())
        assert h.hexdigest()==summary["pairs"][condition]["batch_order_sha256"]
    result={"seed":seed,"success":True,"S_complete":complete,
            "global_first_threshold_verified":True,"cpu_checkpoint_checks":checkpoint_checks,
            "exact_validation_reproductions":validation_checks,"independent_direction_path_checks":direction_checks,
            "old_optimizer_states_preserved":True,"new_optimizer_state_independent":True,
            "first_old_updates_and_task_gradients_bit_identical":True,"same_sampler_path_verified":True,
            "packed_33_neuron_max_logit_error":max_packed_error,
            "previous_experiments_unchanged":True,"training_performed_by_verifier":False}
    result["overlapping_control_paths_bit_identical"] = True
    audit_path=directory/"verification.json"
    if audit_path.exists():
        assert json_read(audit_path)==result, "Reverification differs from original audit"
    else:
        json_write(audit_path,result)
    print(f"VERIFIED seed={seed}: {checkpoint_checks} checkpoints, {direction_checks} step/layer checks, packed max error={max_packed_error:g}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    verify(args.seed,args.run_id)
