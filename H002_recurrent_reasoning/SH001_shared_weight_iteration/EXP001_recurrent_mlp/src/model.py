"""One shared 64 -> 128 -> 64 mapping, with full BPTT."""
import torch
from torch import nn


class RecurrentMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.Sigmoid())

    def forward(self, x, steps=8):
        if x.shape[-1] != 64 or steps < 1:
            raise ValueError('Expected (..., 64) input and positive steps')
        states = []
        for _ in range(steps):
            x = self.net(x)
            states.append(x)
        return states
