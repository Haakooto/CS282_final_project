from BayesLinear import Linear as BayesLinear
from torch import nn
import torch

class FullyConnected(nn.Module):
    # Simple frequentist neural network
    def __init__(self, *, features, classes, hiddens, nonlin="ReLU", use_bias=True):
        super().__init__()

        self.act = eval(f"nn.{nonlin}")
        nodes = [features] + hiddens + [classes]

        layers = []
        for prev, next in zip(nodes[:-2], nodes[1:-1]):
            layers += [nn.Linear(prev, next, bias=use_bias, dtype=torch.float), self.act()]

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
    def __init__(self, *, features, classes, hiddens, nonlin="ReLU", use_bias=True, prior=None):
        super().__init__()

        self.total_kl_div = 0
        self.unfrozen = True

        self.act = eval(f"nn.{nonlin}")  # activation function
        nodes = [features] + hiddens + [classes]

        layers = []
        for prev, next in zip(nodes[:-2], nodes[1:-1]):
            layers += [BayesLinear(inn=prev,
                                   out=next,
                                   prior=prior,
                                   kl_func=self.add_kl_div,
                                   use_bias=True,
                                   ),
                       self.act(),
                       ]

        self.layers = nn.Sequential(*layers,
                                    BayesLinear(inn=nodes[-2],
                                                out=nodes[-1],
                                                prior=prior,
                                                kl_func=self.add_kl_div,
                                                use_bias=True,
                                                ),
                                    )

    def forward(self, x, test=False):
        for layer in self.layers:
            x = layer(x)
            # if test:
            #     print(layer.__class__.__name__, x.mean().data, x.std().data)
        return x

    def add_kl_div(self, kl_div):
        """
        This function is passed along to the distribution class,
        so every time it resamples, it adds to the total kl_div
        by calling this.
        """
        self.total_kl_div += kl_div

    def kl_reset(self, batches=1):
        """
        Retrns the kl_div and sets it to zero
        """
        tmp = self.total_kl_div * self.unfrozen
        self.total_kl_div = 0
        return tmp / batches

    def freeze(self):
        self.unfrozen = False
        for layer in self.layers:
            if isinstance(layer, BayesLinear):
                layer.sample.unfrozen = False


BFC = BayesFullyConnected
FC = FullyConnected
