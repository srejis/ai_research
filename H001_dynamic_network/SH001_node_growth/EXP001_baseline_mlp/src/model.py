import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden1=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
