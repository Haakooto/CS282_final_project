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
        for layer in self.layers:
            x = layer(x)

        kl = 0
        for layer in self.layers:
            if hasattr(layer, "kl_loss"):
                kl += layer.kl_loss()

        return x, kl

    def freeze(self):
        for layer in self.layers:
            if hasattr(layer, "freeze"):
                layer.freeze()

    def unfreeze(self):
        for layer in self.layers:
            if hasattr(layer, "unfreeze"):
                layer.unfreeze()
