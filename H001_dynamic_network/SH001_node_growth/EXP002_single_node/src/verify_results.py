"""Independent read-only numerical audit from persisted branch checkpoints."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import numpy as np
import torch
from torchvision.datasets import MNIST

from experiment_io import ROOT, BASELINE, read_json, write_json, load_checkpoint, sha256
sys.path.insert(0, str(BASELINE / "src"))
from model import MLP
from train_baseline import setup, evaluate, finite_tree


def raw_vectors(state):
    # Independently spell out coordinates instead of reusing analysis slicing.
    result = {
        "fc1.weight": state["fc1.weight"][:256, :784].numpy().astype(np.float64).ravel(),
        "fc1.bias": state["fc1.bias"][:256].numpy().astype(np.float64).ravel(),
        "fc2.weight": state["fc2.weight"][:128, :256].numpy().astype(np.float64).ravel(),
        "fc2.bias": state["fc2.bias"].numpy().astype(np.float64).ravel(),
        "fc3.weight": state["fc3.weight"].numpy().astype(np.float64).ravel(),
        "fc3.bias": state["fc3.bias"].numpy().astype(np.float64).ravel()}
    if state["fc1.weight"].shape[0] == 257:
        result["new.input_weight"] = state["fc1.weight"][256].numpy().astype(np.float64).ravel()
        result["new.input_bias"] = state["fc1.bias"][256:257].numpy().astype(np.float64).ravel()
        result["new.output_weight"] = state["fc2.weight"][:, 256].numpy().astype(np.float64).ravel()
    for layer in ("fc1", "fc2", "fc3"):
        result[f"old.{layer}"] = np.concatenate([result[f"{layer}.weight"], result[f"{layer}.bias"]])
    result["old.all"] = np.concatenate([result[f"old.{layer}"] for layer in ("fc1", "fc2", "fc3")])
    return result


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write-audit", action="store_true")
    args = parser.parse_args()
    reports = ROOT / "results/reports" / args.run_id
    summary = read_json(reports / "summary.json")
    for filename, digest in read_json(reports / "protected_baseline_before.json").items():
        assert sha256(filename) == digest, filename
    for relative, digest in read_json(reports / "artifact_sha256.json").items():
        assert sha256(ROOT / relative) == digest, relative
    setup(42)
    sources = read_json(ROOT / "configs/sources.json")
    baseline_run = ROOT / sources["baseline_results"]
    cfg = read_json(baseline_run / "config.json")
    data_root = ROOT / sources["shared_data"]
    split = torch.load(data_root / "mnist_split_seed42.pt", map_location="cpu", weights_only=True)
    data = MNIST(data_root, train=True, download=False)
    indices = split["validation"]
    x = (data.data[indices].cuda().float().flatten(1) / 255 - cfg["normalization_mean"]) / cfg["normalization_std"]
    y = data.targets[indices].cuda()
    changes = csv_rows(Path(summary["weights_directory"]) / "parameter_changes.csv")
    paired_rows = csv_rows(Path(summary["weights_directory"]) / "paired_changes.csv")
    checkpoints, restored = {}, []
    cp_count, measured = 0, 0
    for branch, folder in summary["checkpoint_directories"].items():
        checkpoints[branch] = {}
        for path in sorted(Path(folder).glob("*.pt")):
            cp = load_checkpoint(path, "cpu")
            assert all(value.device.type == "cpu" for value in cp["model_state_dict"].values())
            assert finite_tree(cp)
            model = MLP(hidden1=cp["architecture"][1]).cuda()
            model.load_state_dict(cp["model_state_dict"], strict=True)
            actual = evaluate(model, x, y, cfg["batch_size"])
            assert actual == cp["validation_evaluation"], (path, actual)
            checkpoints[branch][cp["stage"]] = cp
            cp_count += 1
            restored.append({"branch": branch, "stage": cp["stage"], "validation": actual, "exact_match": True})
        for stage, cp in checkpoints[branch].items():
            ref_stage = "before_insertion" if stage == "before_insertion" else "after_insertion"
            reference = raw_vectors(checkpoints[branch][ref_stage]["model_state_dict"])
            current = raw_vectors(cp["model_state_dict"])
            arrays = np.load(Path(summary["weights_directory"]) / f"{branch}_{stage}.npz")
            for row in (r for r in changes if r["branch"] == branch and r["stage"] == stage):
                name = row["parameter"]
                delta = current[name] - reference[name]
                norm = np.linalg.norm(delta)
                assert np.isclose(float(row["absolute_l2"]), norm, atol=1e-13, rtol=1e-11)
                assert np.isclose(float(row["relative_l2"]), norm / (np.linalg.norm(reference[name]) + 1e-12), atol=1e-13, rtol=1e-11)
                assert np.isclose(float(row["mean_absolute"]), np.abs(delta).mean(), atol=1e-14, rtol=1e-11)
                assert np.isclose(float(row["max_absolute"]), np.abs(delta).max(), atol=1e-14, rtol=1e-11)
                if not name.startswith("old."):
                    assert np.array_equal(arrays[name + ".delta"], delta)
                    direction = delta / norm if norm else np.zeros_like(delta)
                    assert np.allclose(arrays[name + ".direction"], direction, atol=1e-13, rtol=1e-11)
                    counts, edges = np.histogram(delta, bins=50)
                    assert np.array_equal(counts, arrays[name + ".histogram_counts"])
                    assert np.array_equal(edges, arrays[name + ".histogram_edges"])
                measured += 1
        end = checkpoints[branch][f"epoch_{summary['K']:03d}"]
        assert end["branch_step"] == summary["K"] * 430
        assert all(int(s["step"].item()) == end["branch_step"] for s in end["optimizer_state_dict"]["state"].values())
    for row in paired_rows:
        growth, control = row["pair"].split("-")
        stage, name = row["stage"], row["parameter"]
        ref = raw_vectors(checkpoints[control]["before_insertion"]["model_state_dict"])[name]
        dc = raw_vectors(checkpoints[control][stage]["model_state_dict"])[name] - ref
        dg = raw_vectors(checkpoints[growth][stage]["model_state_dict"])[name] - ref
        diff = dg - dc
        assert np.isclose(float(row["absolute_l2"]), np.linalg.norm(diff), atol=1e-13, rtol=1e-11)
        if not name.startswith("old."):
            arrays = np.load(Path(summary["weights_directory"]) / f"{row['pair']}_{stage}.npz")
            assert np.array_equal(arrays[name + ".delta_difference"], diff)
    # B0 starts before any optimizer steps, so it should reproduce EXP001 exactly.
    baseline_complete = load_checkpoint(ROOT / sources["checkpoints"]["complete"])
    b0 = checkpoints["B0"][f"epoch_{summary['K']:03d}"]
    b0_equal = all(torch.equal(b0["model_state_dict"][name], value) for name, value in baseline_complete["model_state_dict"].items())
    assert b0_equal
    for branch in ("B", "C"):
        grad = load_checkpoint(Path(summary["weights_directory"]) / f"{branch}_first_backward_gradients.pt")
        assert torch.count_nonzero(grad["fc1.weight"][256]) == 0
        assert grad["fc1.bias"][256] == 0
        assert grad["fc2.weight"][:, 256].norm() > 0
        assert checkpoints[branch]["after_insertion"]["architecture"] == [784, 257, 128, 10]
    audit = {"success": True, "cpu_checkpoint_loads": cp_count,
             "validation_restore_checks": restored, "independent_numpy_change_checks": measured,
             "paired_change_checks": len(paired_rows), "baseline_all_files_unchanged": True,
             "B0_exactly_reproduces_EXP001_complete_weights": b0_equal,
             "first_input_gradients_zero_and_output_nonzero": True,
             "training_performed_by_verifier": False}
    if args.write_audit:
        write_json(reports / "verification.json", audit)
    print(f"PASS: {cp_count} CPU-loaded checkpoints reproduce validation exactly; {measured} independent tensor/layer checks; {len(paired_rows)} pair checks; B0 equals EXP001.")


if __name__ == "__main__":
    main()
