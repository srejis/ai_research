"""MNIST stage 1: deterministic FP32 training, immutable checkpoints, audit."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.datasets import MNIST

from model import MLP
from paths import DATA_ROOT, experiment_path, checkpoint_directory

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "shuffle_generator": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["shuffle_generator"])


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


@torch.no_grad()
def evaluate(model, x, y, batch_size):
    model.eval()
    total_loss, correct = 0.0, 0
    digest = hashlib.sha256()
    for start in range(0, len(y), batch_size):
        logits = model(x[start:start + batch_size])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        target = y[start:start + batch_size]
        total_loss += nn.functional.cross_entropy(logits, target, reduction="sum").item()
        correct += (logits.argmax(1) == target).sum().item()
        digest.update(logits.cpu().numpy().tobytes())
    return {"loss": total_loss / len(y), "accuracy": correct / len(y),
            "correct": correct, "count": len(y), "logits_sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(experiment_path(args.config).read_text(encoding="utf-8"))
    run = experiment_path(args.run_dir)
    checkpoints = checkpoint_directory(run)
    if checkpoints.exists():
        raise FileExistsError(checkpoints)
    run.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior experiment.
    checkpoints.mkdir(parents=True, exist_ok=False)
    (run / "source").mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, run / "source" / source.name)
    write_json(run / "config.json", cfg)
    setup(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA experiment; CUDA is unavailable")
    device = torch.device("cuda")
    environment = {"timestamp_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                   "python_executable": sys.executable, "python": sys.version,
                   "platform": platform.platform(), "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "numpy": np.__version__,
                   "matplotlib": matplotlib.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "deterministic_algorithms": True,
                   "tf32": False, "autocast": False, "data_order": "CPU randperm per epoch; dedicated generator seeded 42",
                   "data_location": "GPU resident FP32; sequential batch indexing"}
    write_json(run / "environment.json", environment)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run / "requirements-freeze.txt").write_text(freeze, encoding="utf-8")
    train = MNIST(DATA_ROOT, train=True, download=True)
    test = MNIST(DATA_ROOT, train=False, download=True)
    split_path = DATA_ROOT / "mnist_split_seed42.pt"
    split_gen = torch.Generator().manual_seed(cfg["seed"])
    indices = torch.randperm(len(train), generator=split_gen)
    expected = {"train": indices[:55000], "validation": indices[55000:]}
    if split_path.exists():
        split = torch.load(split_path, weights_only=True)
        if not all(torch.equal(split[k], expected[k]) for k in expected):
            raise ValueError("Existing split differs from the fixed seed-42 split")
    else:
        with split_path.open("xb") as stream:
            torch.save(expected, stream)
        split = expected
    shutil.copy2(split_path, run / "data_split.pt")
    split_info = {"path": str(split_path), "sha256": sha256(split_path),
                  "train_count": len(split["train"]), "validation_count": len(split["validation"]),
                  "test_count": len(test), "seed": cfg["seed"],
                  "dataset_files": {str(p.relative_to(DATA_ROOT)): sha256(p)
                                    for p in (DATA_ROOT / "MNIST/raw").glob("*") if p.is_file()},
                  "transform": f"(uint8 / 255 - {cfg['normalization_mean']}) / {cfg['normalization_std']}"}
    write_json(run / "data_manifest.json", split_info)
    def transform(raw):
        return (raw.to(device=device, dtype=torch.float32).flatten(1) / 255 - cfg["normalization_mean"]) / cfg["normalization_std"]
    all_x, all_y = transform(train.data), train.targets.to(device)
    train_x, train_y = all_x[split["train"]], all_y[split["train"]]
    val_x, val_y = all_x[split["validation"]], all_y[split["validation"]]
    test_x, test_y = transform(test.data), test.targets.to(device)
    del all_x, all_y
    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"], foreach=False, fused=False)
    shuffle = torch.Generator().manual_seed(cfg["seed"])
    step, rows = 0, []
    complete = None
    fields = ["epoch", "global_step", "train_loss", "train_accuracy", "validation_loss",
              "validation_accuracy", "optimization_loss", "optimization_accuracy", "learning_rate"]
    with (run / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(cfg["max_epochs"] + 1):
            online_loss, online_correct = 0.0, 0
            if epoch:
                model.train()
                order = torch.randperm(len(train_y), generator=shuffle).to(device)
                for start in range(0, len(order), cfg["batch_size"]):
                    batch = order[start:start + cfg["batch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(train_x[batch])
                    loss = nn.functional.cross_entropy(logits, train_y[batch])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    loss.backward()
                    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError("Non-finite or missing gradients")
                    optimizer.step()
                    step += 1
                    online_loss += loss.item() * len(batch)
                    online_correct += (logits.argmax(1) == train_y[batch]).sum().item()
            training = evaluate(model, train_x, train_y, cfg["batch_size"])
            validation = evaluate(model, val_x, val_y, cfg["batch_size"])
            row = {"epoch": epoch, "global_step": step, "train_loss": training["loss"],
                   "train_accuracy": training["accuracy"], "validation_loss": validation["loss"],
                   "validation_accuracy": validation["accuracy"],
                   "optimization_loss": online_loss / len(train_y) if epoch else None,
                   "optimization_accuracy": online_correct / len(train_y) if epoch else None,
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            checkpoint = {"schema_version": 1, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                          "global_step": step, "seed": cfg["seed"], "rng_state": rng_state(shuffle),
                          "config": cfg, "metrics": row, "train_evaluation": training,
                          "validation_evaluation": validation, "data_split": split_info,
                          "environment": environment}
            if not finite_tree(checkpoint):
                raise FloatingPointError("Non-finite checkpoint state")
            with (checkpoints / f"epoch_{epoch:03d}.pt").open("xb") as file:
                torch.save(checkpoint, file)
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if epoch > 0 and validation["accuracy"] >= cfg["target_accuracy"]:
                complete = epoch
                break
    if complete is None:
        write_json(run / "failure.json", {"success": False, "reason": "95% target not reached before maximum epochs"})
        raise RuntimeError("95% target not reached; run is unsuccessful")
    mid = complete // 2
    checks = []
    # Audit every saved tensor, then reproduce the selected validation logits exactly.
    for epoch in range(complete + 1):
        path = checkpoints / f"epoch_{epoch:03d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert finite_tree(checkpoint), path
        assert checkpoint["epoch"] == epoch
        assert checkpoint["global_step"] == epoch * math.ceil(55000 / cfg["batch_size"])
        assert checkpoint["data_split"]["sha256"] == split_info["sha256"]
        if epoch in {mid, complete}:
            restored = MLP().to(device)
            restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), foreach=False, fused=False)
            restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            restore_rng(checkpoint["rng_state"], shuffle)
            actual = evaluate(restored, val_x, val_y, cfg["batch_size"])
            assert actual == checkpoint["validation_evaluation"], (actual, checkpoint["validation_evaluation"])
            checks.append({"epoch": epoch, "checkpoint": str(path), "sha256": sha256(path),
                           "exact_validation_match": True, "optimizer_restored": True,
                           "rng_restored": True, "validation": actual})
    # The original model is still the final trained model; test set is used once here.
    final_test = evaluate(model, test_x, test_y, cfg["batch_size"])
    assert all(r["validation_accuracy"] < cfg["target_accuracy"] for r in rows[1:-1])
    assert rows[-1]["train_loss"] < rows[0]["train_loss"]
    for metric, title in [("accuracy", "Accuracy"), ("loss", "Cross-entropy loss")]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for prefix, label in [("train", "Train (epoch-end evaluation)"), ("validation", "Validation")]:
            ax.plot([r["epoch"] for r in rows], [r[f"{prefix}_{metric}"] for r in rows], marker="o", label=label)
        if metric == "accuracy":
            ax.axhline(0.95, color="gray", linestyle="--", label="Completion threshold (95%)")
        ax.axvline(mid, color="green", linestyle=":", label=f"E_mid={mid}")
        ax.axvline(complete, color="purple", linestyle=":", label=f"E_complete={complete}")
        ax.set(xlabel="Epoch (0 = before training)", ylabel=title, title=f"MNIST baseline MLP — {title}")
        ax.set_xticks(range(complete + 1))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run / f"{metric}.png", dpi=160)
        plt.close(fig)
    summary = {"success": True, "E_complete": complete, "E_mid": mid, "K": complete - mid,
               "global_step": step, "final_metrics": rows[-1], "test": final_test,
               "restore_checks": checks, "all_checkpoints_finite": True,
               "checkpoint_count": complete + 1, "stage2_executed": False,
               "loss_decreased": rows[-1]["train_loss"] < rows[0]["train_loss"]}
    write_json(run / "summary.json", summary)
    write_json(run / "stage2_plan.json", {"status": "NOT_EXECUTED_REQUIRES_BASELINE_REVIEW",
        "K": complete - mid, "branches": {"B0": {"epoch": mid, "add_neuron": False},
        "B": {"epoch": mid, "add_neuron": True}, "C0": {"epoch": complete, "add_neuron": False},
        "C": {"epoch": complete, "add_neuron": True}}, "optimizer": "fresh AdamW for every branch",
        "paired_order": "Use identical dedicated CPU generator state for each pair; persist every batch index",
        "record_at": ["before insertion", "after insertion", "after first backward (gradients)", "after first update", "every epoch end"],
        "analysis": ["absolute/relative L2 by original parameter indices", "unit update direction and paired cosine", "separate new-neuron parameters", "loss and accuracy"],
        "caution": "Zero output column blocks the first input task gradient; AdamW decay may still change input weights. Fresh optimizer differs from continued baseline moments."})
    report = f"""MNIST 기준 MLP 검증 보고서

성공: 검증 정확도가 최초로 95% 이상이 된 학습 에포크 = {complete}
E_complete = {complete}, E_mid = {mid}, K = {complete-mid}
학습 스텝: {step}; 체크포인트: Epoch 0부터 {complete}까지 {complete+1}개
최종 훈련 정확도: {rows[-1]['train_accuracy']:.4%}
최종 검증 정확도: {rows[-1]['validation_accuracy']:.4%}
최종 테스트 정확도: {final_test['accuracy']:.4%} ({final_test['correct']}/{final_test['count']})
최종 테스트 손실: {final_test['loss']:.8f}
훈련 손실: {rows[0]['train_loss']:.8f} → {rows[-1]['train_loss']:.8f}

검증 결과
- 모든 학습 배치에서 손실과 그래디언트 유한성 검사 통과.
- 모든 에포크 체크포인트의 모델/옵티마이저 텐서 유한성 검사 통과.
- E_mid와 E_complete 모델/옵티마이저/RNG 복원 성공.
- 두 시점의 검증 손실/정확도 및 전체 logits SHA-256이 저장 당시와 정확히 일치.
- 테스트 데이터는 종료 후 1회 평가했으며 완료 판단에 사용하지 않음.
- Epoch 0도 전체 훈련/검증 데이터 평가 후 저장. CSV의 train_*는 에포크 종료 평가,
  optimization_*는 해당 에포크 업데이트 중 관측한 평균이며 Epoch 0에는 비어 있음.
- 완료는 연구의 운영적 정의이며 최적화 수렴 또는 최적 성능을 의미하지 않음.
- E_mid가 0이면 중간 분기는 학습 전 초기 모델을 의미함.
- 재현성 검증은 현재 하드웨어/소프트웨어 내 검사이며 다른 버전/장치의 비트 일치는 보장하지 않음.

경로
실험: {run}
중간 체크포인트: {checkpoints / f'epoch_{mid:03d}.pt'}
완료 체크포인트: {checkpoints / f'epoch_{complete:03d}.pt'}
로그: metrics.csv / 그래프: accuracy.png, loss.png
상세 복원 결과: summary.json
후속 실험: stage2_plan.json 및 source/future_expansion.py (학습 미실행)
"""
    (run / "report_ko.txt").write_text(report, encoding="utf-8")
    manifest = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*") if p.is_file()}
    manifest.update({str(Path("checkpoints") / p.name): sha256(p) for p in checkpoints.glob("*.pt")})
    write_json(run / "artifact_sha256.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
