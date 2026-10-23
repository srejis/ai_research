"""Deterministic state, batch cursors, CPU-portable checkpoints and lossless paths."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def rng_capture():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def rng_restore(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return value


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("xb") as stream:
        torch.save(cpu(value), stream)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(x) for x in value)
    return True


class BatchStream:
    def __init__(self, count, seed, state=None):
        self.count = count
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor, self.samples = 0, 0, 0
        self.order = torch.randperm(count, generator=self.generator)
        if state is not None:
            self.generator.set_state(state["generator"])
            self.epoch, self.cursor, self.samples = state["epoch"], state["cursor"], state["samples"]
            self.order = state["order"].clone()

    def next(self, size):
        if self.cursor == self.count:
            self.order = torch.randperm(self.count, generator=self.generator)
            self.cursor = 0
            self.epoch += 1
        indices = self.order[self.cursor:min(self.cursor + size, self.count)]
        self.cursor += len(indices)
        self.samples += len(indices)
        return indices

    def state(self):
        return {"generator": self.generator.get_state().clone(), "epoch": self.epoch,
                "cursor": self.cursor, "samples": self.samples, "order": self.order.clone()}


def data(cfg, device):
    directory = (ROOT / cfg["shared_data"]).resolve()
    split_path = directory / cfg["fixed_split"]
    split = torch.load(split_path, map_location="cpu", weights_only=True)
    assert len(split["train"]) == 55000 and len(split["validation"]) == 5000
    assert len(torch.unique(torch.cat([split["train"], split["validation"]]))) == 60000
    # Establish exact continuity with EXP001, without importing or changing it.
    legacy = json_read(ROOT.parent / "EXP001_baseline_mlp/results/mnist_baseline_seed42_20261022/data_manifest.json")
    assert digest(split_path) == legacy["sha256"]
    original_cfg = json_read(ROOT.parent / "EXP001_baseline_mlp/configs/baseline.json")
    for key in ("normalization_mean", "normalization_std", "batch_size"):
        assert cfg[key] == original_cfg[key]
    for relative, expected in legacy["dataset_files"].items():
        assert digest(directory / relative) == expected
    mnist = MNIST(directory, train=True, download=False)
    x = (mnist.data.to(device).float().flatten(1) / 255 - cfg["normalization_mean"]) / cfg["normalization_std"]
    y = mnist.targets.to(device)
    return (x[split["train"]], y[split["train"]], x[split["validation"]], y[split["validation"]]), {
        "split_sha256": digest(split_path), "source_manifest": legacy,
        "transform": f"(pixel/255-{cfg['normalization_mean']})/{cfg['normalization_std']}"}


@torch.no_grad()
def evaluate(model, x, y, batch_size=5000):
    model.eval()
    loss, correct = 0.0, 0
    h = hashlib.sha256()
    for offset in range(0, len(y), batch_size):
        logits = model(x[offset:offset+batch_size])
        assert torch.isfinite(logits).all()
        target = y[offset:offset+batch_size]
        loss += nn.functional.cross_entropy(logits, target, reduction="sum").item()
        correct += (logits.argmax(1) == target).sum().item()
        h.update(logits.cpu().numpy().tobytes())
    return {"loss": loss / len(y), "accuracy": correct / len(y), "correct": correct,
            "count": len(y), "logits_sha256": h.hexdigest()}


def flatten(model, original_only=False, gradients=False):
    params = model.original_parameters() if original_only else list(model.named_parameters())
    return np.concatenate([(p.grad if gradients else p).detach().cpu().numpy().ravel() for _, p in params]).copy()


def layout(model, original_only=False):
    params = model.original_parameters() if original_only else list(model.named_parameters())
    result, offset = {}, 0
    for name, p in params:
        result[name] = {"start": offset, "end": offset + p.numel(), "shape": list(p.shape)}
        offset += p.numel()
    return result


def assign_flat(model, vector):
    with torch.no_grad():
        offset = 0
        for _, p in model.named_parameters():
            p.copy_(torch.from_numpy(vector[offset:offset+p.numel()].copy()).reshape(p.shape).to(p.device))
            offset += p.numel()
    assert offset == len(vector)


def layer_vectors(vector, mapping):
    values = {}
    for layer in ("fc1", "fc2", "fc3"):
        values[layer] = np.concatenate([vector[item["start"]:item["end"]] for name, item in mapping.items() if name.startswith(layer + ".")]).astype(np.float64)
    values["all"] = np.concatenate(list(values.values()))
    return values


def cosine(a, b, threshold):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= threshold or nb <= threshold:
        return None
    return float(np.clip(np.dot(a, b) / (na * nb), -1, 1))


class CSV:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8", newline="")
        self.writer = None

    def row(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.stream.close()


class XorWriter:
    """Losslessly store model paths without repeating AdamW/RNG every step."""
    def __init__(self, directory, initial, chunk_size=25):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.previous = np.ascontiguousarray(initial, dtype=np.float32).view(np.uint32).copy()
        with (self.directory / "initial.npy").open("xb") as stream:
            np.save(stream, initial)
        self.chunk_size, self.steps, self.rows = chunk_size, [], []

    def append(self, step, vector):
        bits = np.ascontiguousarray(vector, dtype=np.float32).view(np.uint32)
        self.steps.append(step)
        self.rows.append(bits ^ self.previous)
        self.previous = bits.copy()
        if len(self.rows) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        with (self.directory / f"steps_{self.steps[0]:05d}_{self.steps[-1]:05d}.npz").open("xb") as stream:
            np.savez_compressed(stream, steps=np.array(self.steps), xor=np.stack(self.rows))
        self.rows, self.steps = [], []


def xor_path(directory):
    directory = Path(directory)
    bits = np.load(directory / "initial.npy").view(np.uint32).copy()
    yield 0, bits.view(np.float32).copy()
    for path in sorted(directory.glob("steps_*.npz")):
        with np.load(path) as chunk:
            for step, delta in zip(chunk["steps"], chunk["xor"]):
                bits ^= delta
                yield int(step), bits.view(np.float32).copy()


def checkpoint(path, model, opt, stream, step, cfg, provenance, evaluation=None, extra=None):
    value = {"schema": 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
             "growth": model.growth, "step": step, "config": cfg, "rng": rng_capture(),
             "batch_stream": stream.state(), "layout": layout(model), "original_layout": layout(model, True),
             "provenance": provenance, "validation": evaluation, "extra": extra or {}}
    assert finite(value)
    save(path, value)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_step(model, opt, x, y, indices, device):
    model.train()
    indices_gpu = indices.to(device)
    opt.zero_grad(set_to_none=True)
    sync(device)
    wall_start = time.perf_counter()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if torch.device(device).type == "cuda" else None
    if events:
        events[0].record()
    logits = model(x[indices_gpu])
    loss = nn.functional.cross_entropy(logits, y[indices_gpu])
    loss.backward()
    if events:
        events[1].record()
    sync(device)
    fwd_back_wall = time.perf_counter() - wall_start
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    gradients = flatten(model, True, gradients=True)
    new = {}
    if model.growth:
        activation, contribution, z = model.observation
        new = {"input_gradient_norm": model.new_in.weight.grad.norm().item(),
               "input_bias_gradient": model.new_in.bias.grad.item(),
               "output_gradient_norm": model.new_out.grad.norm().item(),
               "relu_active_fraction": (activation > 0).float().mean().item(),
               "contribution_rms": contribution.square().mean().sqrt().item(),
               "contribution_relative_norm": (contribution.norm() / (z.norm()+1e-12)).item()}
    wall_start = time.perf_counter()
    if events:
        events[2].record()
    opt.step()
    if events:
        events[3].record()
    sync(device)
    optimizer_wall = time.perf_counter() - wall_start
    if model.growth:
        new.update(input_weight_norm=model.new_in.weight.norm().item(), input_bias=model.new_in.bias.item(),
                   output_weight_norm=model.new_out.norm().item())
    return {"train_loss": loss.item(), "train_batch_accuracy": (logits.argmax(1) == y[indices_gpu]).float().mean().item(),
            "gradient": gradients, "new": new, "train_wall_seconds": fwd_back_wall + optimizer_wall,
            "gpu_event_seconds": (events[0].elapsed_time(events[1]) + events[2].elapsed_time(events[3])) / 1000 if events else None}
