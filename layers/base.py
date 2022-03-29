import torch
from torch import nn
import torch.functional as F


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for module in self.modules():
            x = module(x)

        return x