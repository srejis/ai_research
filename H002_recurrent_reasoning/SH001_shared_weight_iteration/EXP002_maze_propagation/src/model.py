"""A single shared MLP; no search, grid adjacency, or target access."""
import torch
from torch import nn


class MazeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(192, 256), nn.ReLU(),
                                  nn.Linear(256, 192), nn.Sigmoid())

    def forward(self, walls, goal, steps=8, mode='C'):
        if walls.shape != goal.shape or walls.shape[-1] != 64:
            raise ValueError('Walls and goal must both have shape (batch, 64)')
        if mode not in 'ABCD' or steps < 1:
            raise ValueError('Invalid mode or recurrence length')
        state = torch.cat((walls, goal, goal), dim=-1)
        outputs = []
        for _ in range(steps):
            state = self.core(state)
            outputs.append(state)
            if mode in 'CD':
                state = torch.cat((walls, goal, state[..., 128:]), dim=-1)
        return outputs
