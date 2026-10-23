"""Portable CPU checkpoints and isolated optimizer policy for EXP002."""
import copy
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "EXP001_baseline_mlp"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def load_checkpoint(path, device="cpu"):
    """Files created by this project only; Python/NumPy RNG objects are included."""
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path, value):
    with Path(path).open("xb") as stream:
        torch.save(cpu_copy(value), stream)


def make_optimizer(model, cfg, policy="fresh"):
    # Keeping this policy explicit allows a separate state-migration experiment.
    if policy != "fresh":
        raise NotImplementedError("Optimizer-state migration is outside this experiment")
    return torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"], foreach=False, fused=False)
