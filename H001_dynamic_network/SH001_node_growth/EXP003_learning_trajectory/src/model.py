"""Exact original-coordinate preservation with a separate one-neuron path."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class TinyMLP(nn.Module):
    def __init__(self, growth=False, new_seed=42042):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        self.growth = growth
        if growth:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(new_seed)
                self.new_in = nn.Linear(784, 1)
            self.new_out = nn.Parameter(torch.zeros(16, 1))
        self.observation = None

    def forward(self, x):
        x = x.flatten(1)
        z = self.fc2(F.relu(self.fc1(x)))
        if self.growth:
            h = F.relu(self.new_in(x))
            contribution = F.linear(h, self.new_out)
            # Detached statistics are read before validation can replace them.
            self.observation = (h.detach(), contribution.detach(), z.detach())
            z = z + contribution
        return self.fc3(F.relu(z))

    def original_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if n.startswith("fc")]

    def new_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if not n.startswith("fc")]


def clone_model(source_state, growth=False, new_seed=42042, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        model = TinyMLP(growth, new_seed)
    result = model.load_state_dict(source_state, strict=not growth)
    if growth:
        assert set(result.missing_keys) == {"new_out", "new_in.weight", "new_in.bias"}
        assert not result.unexpected_keys
    return model.to(device)


def optimizer(model, cfg, source_state=None):
    opt = torch.optim.AdamW([p for _, p in model.original_parameters()],
                           lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                           betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
    if source_state is not None:
        # AdamW's CPU step tensors must not be shared between paired optimizers.
        opt.load_state_dict(copy.deepcopy(source_state))
    if model.growth:
        opt.add_param_group({"params": [p for _, p in model.new_parameters()]})
    return opt
