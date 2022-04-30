from Gaussian import Gauss#, Uniform
from torch import nn
import torch.nn.functional as F
import torch

# Distributions = {"gauss": Gauss, "uniform": Uniform}
Distributions = {"gauss": Gauss}


class Linear(nn.Module):
    def __init__(self, *, inn, out, prior, kl_adder, use_bias=True):
        super().__init__()

        self.use_bias = use_bias

        assert prior, "Please specify distribution!"

        self.prior = prior
        dist = Distributions[prior["dist"]]
        self.sample = dist(inn, out, **self.prior["params"], bias=self.use_bias, kl_adder=kl_adder)

    def forward(self, x):
        weight, bias = self.sample()

        return F.linear(x, weight, bias)
