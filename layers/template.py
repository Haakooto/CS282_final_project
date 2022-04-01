import torch
from torch import nn
import torch.functional as F


class BaseModule(nn.Module):
    """
    Simple layer skeleton. Can be inherited for making full models.
    Should also add calculation of KL-loss, for bayesian stuff. See papers
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for module in self.children():
            x = module(x)

        return x
