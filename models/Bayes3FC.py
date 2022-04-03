# This imports all the functions of all programs in the layers folder.
from layers import *
from torch import nn


class Bayesian3FC(BaseModule):
    """
    Simple implementation of a 3-layered FC Bayesian net with ReLU as non-linearity
    Mainly for easy implementation, more complex models will be made later
    """

    def __init__(self, features=2, n1=2, n2=2, classes=1, prior=None):
        super().__init__()

        # Here we use the BLinear from layers/__init__.py, that was imported.
        fc1 = BLinear(features, n1, prior=prior)
        non_lin1 = nn.ReLU()
        fc2 = BLinear(n1, n2, prior=prior)
        non_lin2 = nn.ReLU()
        fc3 = BLinear(n2, classes, prior=prior)

        self.layers = nn.Sequential(fc1, non_lin1, fc2, non_lin2, fc3)


class BayesianFC(BaseModule):
    """
    Fully connected bayesian model with N layes
    """
    nonlin = {"relu": nn.ReLU, "softplus": nn.Softplus}

    def __init__(self, inputs, outputs, hidden_nodes=[10, 10], nonlinearity="relu", prior=None):
        super().__init__()

        self.act = BayesianFC.nonlin[nonlinearity]
        nodes = [inputs] + hidden_nodes + [outputs]

        layers = []
        for prev, next in zip(nodes[:-2], nodes[1:-1]):
            layers += [BLinear(prev, next, prior=prior), self.act()]

        self.layers = nn.Sequential(
            *layers, BLinear(nodes[-2], nodes[-1], prior=prior))
