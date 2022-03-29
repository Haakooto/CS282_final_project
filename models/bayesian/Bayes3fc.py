from layers.base import BaseModule
from layers.Bayesian.linear import Linear
from torch import nn


class Bayesian3FC(BaseModule):
    def __init__(self, features, n1, n2, n3, classes=1):
        super().__init__()

        self.fc1 = Linear(features, n1)
        self.non_lin1 = nn.ReLU()
        self.fc2 = Linear(n2, n3)
        self.non_lin2 = nn.ReLU()
        self.fc3 = Linear(n3, classes)



