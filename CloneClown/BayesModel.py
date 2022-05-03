from BayesLinear import Linear as BayesLinear
from torch import nn
from torch.nn import functional as F
import torch


class FullyConnected(nn.Module):
    # Simple frequentist neural network
    def __init__(self, *, features, classes, hiddens, nonlin=nn.ReLU, use_bias=True):
        super().__init__()

        self.activation = nonlin
        nodes = [features] + hiddens + [classes]

        layers = []
        for prev, next in zip(nodes[:-2], nodes[1:-1]):
            layers += [nn.Linear(prev, next, bias=use_bias,
                                 dtype=torch.float), self.activation()]

        self.layers = nn.Sequential(*layers,
                                    nn.Linear(
                                        nodes[-2], nodes[-1], bias=use_bias, dtype=torch.float),
                                    )
        self.float()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BayesFullyConnected(nn.Module):
    # simple bayesian neural network
    def __init__(self, *, features, classes, hiddens, nonlin=nn.ReLU, prior=None, dtype=None, device=None):
        super().__init__()

        if device is None:
            self.device = torch.device("cpu")
        if dtype is None:
            self.dtype = torch.double

        self.total_kl_div = 0
        self.unfrozen = True

        if prior is None:
            prior = {}
        prior.setdefault("dist",     "normal")
        prior.setdefault("loc",      0)
        prior.setdefault("scale",    1)
        prior.setdefault("use_bias", True)

        self.activation = nonlin

        nodes = [features] + hiddens + [classes]

        layers = []
        for prev, next in zip(nodes[:-2], nodes[1:-1]):
            layers += [BayesLinear(**prior,
                                   inn=prev,
                                   out=next,
                                   device=self.device,
                                   dtype=self.dtype,
                                   ),
                       self.activation(),
                       ]

        self.layers = nn.Sequential(*layers,
                                    BayesLinear(**prior,
                                                inn=nodes[-2],
                                                out=nodes[-1],
                                                device=self.device,
                                                dtype=self.dtype,
                                                ),
                                    )

    def forward(self, x, train=False):
        for layer in self.layers:
            x = layer(x)

        if train:
            for layer in self.layers:
                if hasattr(layer, "kl_div"):
                    self.total_kl_div += layer.kl_div().sum(-1).mean()
        return x

    def kl_reset(self):
        """
        Retrns the kl_div and resets it to zero
        """
        tmp = self.total_kl_div * self.unfrozen
        self.total_kl_div = 0
        return tmp


BFC = BayesFullyConnected
FC = FullyConnected
