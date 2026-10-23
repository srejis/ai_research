"""Read-only validation reproduction in a new Python process; no training."""
import argparse
from pathlib import Path
import json
import torch
from torchvision.datasets import MNIST
from model import MLP
from train_baseline import setup, evaluate, finite_tree, sha256
from paths import DATA_ROOT, experiment_path, checkpoint_directory, artifact_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional new audit JSON; refuses overwrite")
    args = parser.parse_args()
    run = experiment_path(args.run_dir)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    setup(cfg["seed"])
    manifest = json.loads((run / "artifact_sha256.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        assert sha256(artifact_path(run, relative)) == expected, relative
    data = MNIST(DATA_ROOT, train=True, download=False)
    indices = torch.load(run / "data_split.pt", weights_only=True)
    assert len(indices["train"]) == 55000 and len(indices["validation"]) == 5000
    assert len(torch.unique(torch.cat(list(indices.values())))) == 60000
    val = indices["validation"]
    x = (data.data[val].cuda().float().flatten(1) / 255 - cfg["normalization_mean"]) / cfg["normalization_std"]
    y = data.targets[val].cuda()
    checks = []
    for epoch in sorted({summary["E_mid"], summary["E_complete"]}):
        path = checkpoint_directory(run) / f"epoch_{epoch:03d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        original = next(c for c in summary["restore_checks"] if c["epoch"] == epoch)
        assert sha256(path) == original["sha256"]
        assert finite_tree(checkpoint)
        model = MLP().cuda()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor.cpu(), checkpoint["model_state_dict"][name]), name
        optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restored_state = optimizer.state_dict()
        assert restored_state["param_groups"] == checkpoint["optimizer_state_dict"]["param_groups"]
        for key, state in checkpoint["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                actual_value = restored_state["state"][key][name]
                assert torch.equal(actual_value.cpu(), value) if torch.is_tensor(value) else actual_value == value
        actual = evaluate(model, x, y, cfg["batch_size"])
        assert actual == checkpoint["validation_evaluation"]
        check = {"epoch": epoch, "checkpoint": str(path), "sha256": sha256(path),
                 "exact_match": True, "model_weights_equal": True,
                 "optimizer_state_equal": True, "validation": actual}
        checks.append(check)
        print(json.dumps(check), flush=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump({"training_executed": False, "checks": checks, "artifact_hashes_match": True}, stream, indent=2)
    print("PASS: artifact hashes, split disjointness, fresh-process checkpoint validation. No training performed.")


if __name__ == "__main__":
    main()
