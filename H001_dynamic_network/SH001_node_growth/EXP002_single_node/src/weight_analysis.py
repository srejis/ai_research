"""Original-coordinate deltas, distributions and paired update directions."""
import csv
from pathlib import Path

import numpy as np
import torch


def coordinates(state):
    result = {name: value.detach().cpu().double().clone() for name, value in state.items()}
    expanded = result["fc1.weight"].shape[0] == 257
    if expanded:
        result["new.input_weight"] = result["fc1.weight"][256].clone()
        result["new.input_bias"] = result["fc1.bias"][256:257].clone()
        result["new.output_weight"] = result["fc2.weight"][:, 256].clone()
    result["fc1.weight"] = result["fc1.weight"][:256].clone()
    result["fc1.bias"] = result["fc1.bias"][:256].clone()
    result["fc2.weight"] = result["fc2.weight"][:, :256].clone()
    return result


def vectors(values):
    result = {name: value.flatten() for name, value in values.items()}
    for layer in ("fc1", "fc2", "fc3"):
        result[f"old.{layer}"] = torch.cat([values[f"{layer}.weight"].flatten(), values[f"{layer}.bias"].flatten()])
    result["old.all"] = torch.cat([result[f"old.{layer}"] for layer in ("fc1", "fc2", "fc3")])
    return result


def stats(delta, base, epsilon):
    absolute = delta.abs()
    norm = torch.linalg.vector_norm(delta).item()
    base_norm = torch.linalg.vector_norm(base).item()
    quantiles = torch.quantile(delta, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    return {"absolute_l2": norm, "base_l2": base_norm,
            "relative_l2": norm / (base_norm + epsilon),
            "base_is_zero": base_norm == 0, "current_l2": torch.linalg.vector_norm(base + delta).item(),
            "mean_absolute": absolute.mean().item(), "max_absolute": absolute.max().item(),
            "q0": quantiles[0].item(), "q25": quantiles[1].item(), "q50": quantiles[2].item(),
            "q75": quantiles[3].item(), "q100": quantiles[4].item(), "numel": delta.numel()}


def measure(start, current, epsilon, metadata):
    rows, arrays = [], {}
    before, after = vectors(start), vectors(current)
    for name, base in before.items():
        delta = after[name] - base
        values = stats(delta, base, epsilon)
        rows.append({**metadata, "parameter": name, **values})
        if not name.startswith("old."):
            arrays[name + ".delta"] = delta.numpy()
            arrays[name + ".direction"] = (delta / values["absolute_l2"] if values["absolute_l2"] else torch.zeros_like(delta)).numpy()
            hist, bins = np.histogram(delta.numpy(), bins=50)
            arrays[name + ".histogram_counts"] = hist
            arrays[name + ".histogram_edges"] = bins
    return rows, arrays


def paired(start, control, growth, epsilon, metadata):
    rows, arrays = [], {}
    baseline = vectors(start)
    c, g = vectors(control), vectors(growth)
    for name, base in baseline.items():
        if name.startswith("new."):
            continue
        dc, dg = c[name] - base, g[name] - base
        diff = dg - dc
        nc, ng = torch.linalg.vector_norm(dc).item(), torch.linalg.vector_norm(dg).item()
        cosine = torch.dot(dc, dg).item() / (nc * ng) if nc and ng else None
        rows.append({**metadata, "parameter": name, **stats(diff, base, epsilon),
                     "control_delta_l2": nc, "growth_delta_l2": ng,
                     "growth_minus_control_delta_norm": ng - nc,
                     "update_cosine": max(-1.0, min(1.0, cosine)) if cosine is not None else None})
        if not name.startswith("old."):
            arrays[name + ".delta_difference"] = diff.numpy()
    return rows, arrays


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
