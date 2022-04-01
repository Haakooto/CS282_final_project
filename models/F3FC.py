from layers import *
from torch import nn


class F3FC(BaseModule):
    """
    Simple implementation of a 3-layered frequentist FC net with ReLU as non-linearity
    Mainly for easy implementation, more complex models will be made later
    """
    def __init__(self, features=2, n1=2, n2=2, n3=2, classes=1):
        super().__init__()

        self.fc1       = nn.Linear(features, n1)
        self.non_lin1  = nn.PReLU()
        self.fc2       = nn.Linear(n2, n3)
        self.non_lin2  = nn.ReLU()
        self.fc3       = nn.Linear(n3, classes)
