"""Stage 2 building blocks only. No training entry point and no automatic execution."""
import torch
import sys
from pathlib import Path

# Reuse EXP001's model definition without copying baseline weights or code.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EXP001_baseline_mlp/src"))
from model import MLP


def add_one_neuron(model, seed=42001):
    """Append index 256; preserve all original connections and initial logits.

    The new fc2 column is zero, so the first backward pass gives zero task-loss
    gradient to the new fc1 row/bias. AdamW decay can still change that row.
    """
    if model.fc1.out_features != 256:
        raise ValueError("Expected the baseline 784-256-128-10 model")
    device = next(model.parameters()).device
    # Local CPU RNG scope avoids changing the branch's data-order RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        expanded = MLP(hidden1=257)
    expanded = expanded.to(device=device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        expanded.fc1.weight[:256].copy_(model.fc1.weight)
        expanded.fc1.bias[:256].copy_(model.fc1.bias)
        expanded.fc2.weight[:, :256].copy_(model.fc2.weight)
        expanded.fc2.weight[:, 256].zero_()
        expanded.fc2.bias.copy_(model.fc2.bias)
        expanded.fc3.load_state_dict(model.fc3.state_dict())
    return expanded


def parameter_groups(model):
    """Stable original indices for later displacement/direction measurements."""
    result = {}
    for name, value in model.named_parameters():
        if name == "fc1.weight":
            result[name] = value[:256]
        elif name == "fc1.bias":
            result[name] = value[:256]
        elif name == "fc2.weight":
            result[name] = value[:, :256]
        else:
            result[name] = value
    if model.fc1.out_features == 257:
        result["new.input_weight"] = model.fc1.weight[256]
        result["new.input_bias"] = model.fc1.bias[256:257]
        result["new.output_weight"] = model.fc2.weight[:, 256]
    return result


def change_metrics(before, after):
    a, b = before.detach().double().flatten(), after.detach().double().flatten()
    delta = b - a
    norm = torch.linalg.vector_norm(delta)
    base = torch.linalg.vector_norm(a)
    return {"absolute_l2": norm.item(),
            "relative_l2": (norm / base).item() if base > 0 else None,
            "relative_l2_undefined": bool(base == 0),
            "direction": delta / norm if norm > 0 else torch.zeros_like(delta)}
