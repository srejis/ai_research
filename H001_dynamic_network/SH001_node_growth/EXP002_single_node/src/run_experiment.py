"""Run four paired branches from immutable EXP001 checkpoints; never retrain EXP001."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import math
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST

from experiment_io import ROOT, BASELINE, read_json, write_json, sha256, cpu_copy, load_checkpoint, save_checkpoint, make_optimizer
sys.path.insert(0, str(BASELINE / "src"))
from model import MLP
from train_baseline import setup, evaluate, finite_tree, rng_state
from future_expansion import add_one_neuron, parameter_groups
from weight_analysis import coordinates, measure, paired, write_csv

BRANCHES = {"B0": ("mid", False, "B0_mid_control"), "B": ("mid", True, "B_mid_growth"),
            "C0": ("complete", False, "C0_final_control"), "C": ("complete", True, "C_final_growth")}


@torch.no_grad()
def compare_logits(before, after, x, batch, atol, rtol):
    before.eval()
    after.eval()
    maximum, squared, count, equal = 0.0, 0.0, 0, True
    for offset in range(0, len(x), batch):
        a, b = before(x[offset:offset + batch]), after(x[offset:offset + batch])
        diff = (a - b).double()
        maximum = max(maximum, diff.abs().max().item())
        squared += diff.square().sum().item()
        count += diff.numel()
        equal = equal and bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    return {"allclose": equal, "max_absolute_logit_difference": maximum,
            "rms_logit_difference": math.sqrt(squared / count), "atol": atol, "rtol": rtol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run-id must be one directory name")
    cfg = read_json(ROOT / "configs/experiment.json")
    sources = read_json(ROOT / "configs/sources.json")
    baseline_run = (ROOT / sources["baseline_results"]).resolve()
    config = read_json(baseline_run / "config.json")
    summary = read_json(baseline_run / "summary.json")
    setup(cfg["seed"])
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly if desired")
    # Protect every baseline file (not only the two checkpoints) before any training.
    protected = {str(p): sha256(p) for p in BASELINE.rglob("*") if p.is_file()}
    with (baseline_run / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        baseline_rows = list(csv.DictReader(stream))
    reached = [int(row["epoch"]) for row in baseline_rows if int(row["epoch"]) > 0 and float(row["validation_accuracy"]) >= 0.95]
    if not reached:
        raise RuntimeError("No baseline completion in log")
    complete, mid = reached[0], reached[0] // 2
    k = complete - mid
    assert (complete, mid, k) == (summary["E_complete"], summary["E_mid"], summary["K"])
    assert (complete, mid, k) == (sources["E_complete"], sources["E_mid"], sources["K"])
    paths = {key: (ROOT / value).resolve() for key, value in sources["checkpoints"].items()}
    source_cp = {}
    for key, path in paths.items():
        assert sha256(path) == sources["expected_sha256"][key]
        source_cp[key] = load_checkpoint(path)  # CPU-portable load.
        assert source_cp[key]["config"] == config and finite_tree(source_cp[key])
    # Verify original run artifacts through EXP001's relocated manifest resolver.
    from paths import artifact_path
    for relative, expected in read_json(baseline_run / "artifact_sha256.json").items():
        assert sha256(artifact_path(baseline_run, relative)) == expected, relative
    data_root = (ROOT / sources["shared_data"]).resolve()
    dataset = MNIST(data_root, train=True, download=False)
    split = torch.load(data_root / "mnist_split_seed42.pt", map_location="cpu", weights_only=True)
    assert sha256(data_root / "mnist_split_seed42.pt") == source_cp["mid"]["data_split"]["sha256"]
    assert len(split["train"]) == 55000 and len(split["validation"]) == 5000
    assert len(torch.unique(torch.cat([split["train"], split["validation"]]))) == 60000
    for relative, expected in read_json(baseline_run / "data_manifest.json")["dataset_files"].items():
        assert sha256(data_root / relative) == expected
    x = (dataset.data.to(device).float().flatten(1) / 255 - config["normalization_mean"]) / config["normalization_std"]
    y = dataset.targets.to(device)
    tx, ty = x[split["train"]], y[split["train"]]
    vx, vy = x[split["validation"]], y[split["validation"]]
    del x, y
    batch = config["batch_size"]
    restored, expanded, preflight = {}, {}, {}
    for key, checkpoint in source_cp.items():
        model = MLP().to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        validation = evaluate(model, vx, vy, batch)
        assert validation == checkpoint["validation_evaluation"], (key, validation)
        growth = add_one_neuron(model, cfg["new_neuron_seed"])
        original_groups, growth_groups = parameter_groups(model), parameter_groups(growth)
        for name, value in original_groups.items():
            assert torch.equal(value, growth_groups[name]), name
        assert growth.fc1.out_features == 257 and growth.fc2.in_features == 257
        assert sum(p.numel() for p in growth.parameters()) - sum(p.numel() for p in model.parameters()) == 913
        assert torch.count_nonzero(growth.fc2.weight[:, 256]) == 0
        preserve = compare_logits(model, growth, vx, batch, cfg["logits_atol"], cfg["logits_rtol"])
        if not preserve["allclose"]:
            raise RuntimeError(f"Unexpected insertion logits: {key} {preserve}")
        restored[key], expanded[key] = model, growth
        preflight[key] = {"checkpoint": str(paths[key]), "sha256": sha256(paths[key]),
                          "epoch": checkpoint["epoch"], "validation": validation,
                          "original_coordinates_equal": True, "added_neurons": 1,
                          "added_parameters": 913, "output_preservation": preserve}
    # All restoration/insertion checks passed before training any branch.
    report_dir = ROOT / "results" / "reports" / args.run_id
    metrics_dir = ROOT / "results" / "metrics" / args.run_id
    weight_dir = ROOT / "results" / "weights" / args.run_id
    figure_dir = ROOT / "results" / "figures" / args.run_id
    cp_dirs = {branch: ROOT / "checkpoints" / spec[2] / args.run_id for branch, spec in BRANCHES.items()}
    destinations = [report_dir, metrics_dir, weight_dir, figure_dir, *cp_dirs.values()]
    for path in destinations:
        if path.exists():
            raise FileExistsError(path)
    for path in destinations:
        path.mkdir(parents=True, exist_ok=False)
    snapshot = report_dir / "source"
    shutil.copytree(Path(__file__).parent, snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", report_dir / "configs")
    write_json(report_dir / "baseline_verification.json", preflight)
    write_json(report_dir / "protected_baseline_before.json", protected)
    write_json(report_dir / "parameter_mapping.json", {
        "fc1.weight": "original rows [0:256], columns [0:784]; new row [256]",
        "fc1.bias": "original [0:256]; new [256]",
        "fc2.weight": "original rows [0:128], columns [0:256]; new column [256]",
        "fc2.bias": "all original", "fc3.weight": "all original", "fc3.bias": "all original",
        "relative_change": "L2(delta)/(L2(start)+1e-12)",
        "new_baseline": "after insertion; zero output base makes relative L2 epsilon-dependent; use absolute L2"})
    orders, order_states = {}, {}
    for key, checkpoint in source_cp.items():
        gen = torch.Generator()
        gen.set_state(checkpoint["rng_state"]["shuffle_generator"])
        orders[key], order_states[key] = [], [gen.get_state().clone()]
        for _ in range(k):
            orders[key].append(torch.randperm(len(ty), generator=gen))
            order_states[key].append(gen.get_state().clone())
        with (metrics_dir / f"{key}_batch_order.npz").open("xb") as stream:
            np.savez_compressed(stream, **{f"epoch_{i+1:03d}": order.numpy() for i, order in enumerate(orders[key])})
    performance, changes, gradients, branch_results = [], [], {}, {}
    for branch, (key, growth, _) in BRANCHES.items():
        setup(cfg["seed"])
        model = MLP().to(device)
        model.load_state_dict(source_cp[key]["model_state_dict"])
        optimizer = make_optimizer(model, config, cfg["optimizer_policy"])
        start_old = coordinates(model.state_dict())
        start_epoch = source_cp[key]["epoch"]
        step = 0
        consumed = []
        shuffle = torch.Generator()
        shuffle.set_state(order_states[key][0])

        def record(stage, local_epoch, reference):
            model.eval()
            train_eval = evaluate(model, tx, ty, batch)
            val_eval = evaluate(model, vx, vy, batch)
            metadata = {"branch": branch, "stage": stage, "local_epoch": local_epoch,
                        "source_epoch": start_epoch, "step": step}
            metrics = {**metadata, "train_loss": train_eval["loss"], "train_accuracy": train_eval["accuracy"],
                       "validation_loss": val_eval["loss"], "validation_accuracy": val_eval["accuracy"],
                       "validation_logits_sha256": val_eval["logits_sha256"]}
            performance.append(metrics)
            payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                       "architecture": [784, model.fc1.out_features, 128, 10], "config": config,
                       "experiment_config": cfg, "branch": branch, "stage": stage,
                       "epoch": start_epoch + local_epoch, "local_epoch": local_epoch,
                       "global_step": source_cp[key]["global_step"] + step, "branch_step": step,
                       "source_checkpoint": str(paths[key]), "source_sha256": sources["expected_sha256"][key],
                       "rng_state": rng_state(shuffle), "metrics": metrics,
                       "train_evaluation": train_eval, "validation_evaluation": val_eval,
                       "data_split": source_cp[key]["data_split"],
                       "order_path": str(metrics_dir / f"{key}_batch_order.npz"),
                       "order_sha256": sha256(metrics_dir / f"{key}_batch_order.npz"),
                       "optimizer_policy": "fresh", "precision": "FP32; TF32 disabled"}
            assert finite_tree(payload)
            save_checkpoint(cp_dirs[branch] / f"{stage}.pt", payload)
            rows, arrays = measure(reference, coordinates(model.state_dict()), cfg["relative_change_epsilon"], metadata)
            changes.extend(rows)
            with (weight_dir / f"{branch}_{stage}.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            print(f"{branch} {stage}: step={step}, val_acc={val_eval['accuracy']:.6f}, val_loss={val_eval['loss']:.8f}", flush=True)

        record("before_insertion", 0, start_old)
        if growth:
            model = expanded[key]
        optimizer = make_optimizer(model, config, cfg["optimizer_policy"])
        start = coordinates(model.state_dict())
        record("after_insertion", 0, start)
        for local_epoch, order_cpu in enumerate(orders[key], 1):
            order = order_cpu.to(device)
            online_loss, online_correct = 0.0, 0
            for offset in range(0, len(ty), batch):
                model.train()
                indices = order[offset:offset + batch]
                optimizer.zero_grad(set_to_none=True)
                logits = model(tx[indices])
                loss = nn.functional.cross_entropy(logits, ty[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{branch}: loss")
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                online_loss += loss.item() * len(indices)
                online_correct += (logits.argmax(1) == ty[indices]).sum().item()
                if step == 0:
                    grad = {name: p.grad for name, p in model.named_parameters()}
                    save_checkpoint(weight_dir / f"{branch}_first_backward_gradients.pt", grad)
                    if growth:
                        gradients[branch] = {"new_input_weight_gradient_l2": model.fc1.weight.grad[256].norm().item(),
                            "new_input_bias_gradient": model.fc1.bias.grad[256].item(),
                            "new_output_weight_gradient_l2": model.fc2.weight.grad[:, 256].norm().item()}
                        assert gradients[branch]["new_input_weight_gradient_l2"] == 0
                        assert gradients[branch]["new_input_bias_gradient"] == 0
                optimizer.step()
                step += 1
                if step == 1:
                    record("first_update", 0, start)
                    if growth:
                        delta = model.fc1.weight[256].detach().cpu().double() - start["new.input_weight"]
                        expected = -config["learning_rate"] * config["weight_decay"] * start["new.input_weight"]
                        gradients[branch]["first_input_update_l2"] = delta.norm().item()
                        gradients[branch]["adamw_decay_prediction_l2"] = expected.norm().item()
                        gradients[branch]["decay_prediction_max_residual"] = (delta - expected).abs().max().item()
            shuffle.set_state(order_states[key][local_epoch])
            consumed.append(sha256(metrics_dir / f"{key}_batch_order.npz"))
            record(f"epoch_{local_epoch:03d}", local_epoch, start)
            performance[-1]["optimization_loss"] = online_loss / len(ty)
            performance[-1]["optimization_accuracy"] = online_correct / len(ty)
        assert step == k * math.ceil(len(ty) / batch)
        branch_results[branch] = {"steps": step, "epochs": k, "source_epoch": start_epoch,
                                  "order_sha256": consumed[0], "final_metrics": performance[-1]}
    # Give every performance row the same schema.
    for row in performance:
        row.setdefault("optimization_loss", None)
        row.setdefault("optimization_accuracy", None)
    write_csv(metrics_dir / "performance.csv", performance)
    write_csv(weight_dir / "parameter_changes.csv", changes)
    pair_rows = []
    for control, growth in (("B0", "B"), ("C0", "C")):
        assert branch_results[control]["order_sha256"] == branch_results[growth]["order_sha256"]
        key = BRANCHES[control][0]
        start = coordinates(source_cp[key]["model_state_dict"])
        for stage in ["after_insertion", "first_update", *[f"epoch_{i:03d}" for i in range(1, k+1)]]:
            c = load_checkpoint(cp_dirs[control] / f"{stage}.pt")
            g = load_checkpoint(cp_dirs[growth] / f"{stage}.pt")
            metadata = {"pair": f"{growth}-{control}", "stage": stage,
                        "local_epoch": c["local_epoch"], "step": c["branch_step"]}
            rows, arrays = paired(start, coordinates(c["model_state_dict"]), coordinates(g["model_state_dict"]), cfg["relative_change_epsilon"], metadata)
            pair_rows.extend(rows)
            with (weight_dir / f"{growth}-{control}_{stage}.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
    write_csv(weight_dir / "paired_changes.csv", pair_rows)
    write_json(weight_dir / "first_gradient_analysis.json", gradients)
    for filename, digest in protected.items():
        assert sha256(filename) == digest, filename
    result = {"run_id": args.run_id, "E_mid": mid, "E_complete": complete, "K": k,
              "optimizer_policy": "fresh AdamW in every branch", "device": str(device),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "branches": branch_results, "insertion_checks": preflight, "first_gradients": gradients,
              "baseline_all_files_unchanged": True, "test_set_used": False,
              "checkpoint_directories": {key: str(path) for key, path in cp_dirs.items()},
              "metrics_directory": str(metrics_dir), "weights_directory": str(weight_dir),
              "figure_directory": str(figure_dir), "report_directory": str(report_dir)}
    write_json(report_dir / "summary.json", result)
    from visualize_results import generate
    generate(result, performance, changes, pair_rows)
    files = [*report_dir.rglob("*"), *metrics_dir.rglob("*"), *weight_dir.rglob("*"), *figure_dir.rglob("*")]
    for folder in cp_dirs.values():
        files.extend(folder.rglob("*"))
    manifest = {str(path.relative_to(ROOT)): sha256(path) for path in files if path.is_file()}
    write_json(report_dir / "artifact_sha256.json", manifest)
    print(f"COMPLETE: {report_dir}", flush=True)


if __name__ == "__main__":
    main()
