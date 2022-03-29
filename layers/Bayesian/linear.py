from ..base import BaseModule
from torch import nn
import torch


class Linear(BaseModule):
    def __init__(self, in_nodes, out_nodes, priors=None):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if priors is None: priors = {}
        priors.setdefault("prior_mean", 0)
        priors.setdefault("prior_var", 1)

        self.W_mu = nn.Parameter(torch.empty(out_nodes, in_nodes, device=self.device))
        self.W_sig = nn.Parameter(torch.empty(out_nodes, in_nodes, device=self.device))

        self.b_mu = nn.Parameter(torch.empty(out_nodes, device=self.device))
        self.b_sig = nn.Parameter(torch.empty(out_nodes, device=self.device))

        self.init_params()

    def init_params(self):
        # self.W_mu.data.normal()
        pass

    def forward(self, x):
        return x
