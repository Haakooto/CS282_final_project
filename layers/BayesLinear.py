from zmq import device
from .template import BaseModule
from priors import implemented_dists as Dists
from torch import nn
from torch.nn import functional as F
import torch


"""
Our bayesian neural network linear layer
NOT DONE YET!
The structure is based upon two repos doing BNNs,
I have taken the specific features I like from each of them
Use these two as base as not to have to reinvent wheel.
links to repos:
    https://github.com/Harry24k/bayesian-neural-network-pytorch
        (found by googling 'bayesian neural network pytorch')
    https://github.com/kumar-shridhar/PyTorch-BayesianCNN
        (from paper 'A Comprehensive guide to Bayesian Convolutional Neural Network with Variational Inference' in our google docs)

Still not too sure how to do the prior.
Think we should add a class-template for sampling them, and then
use many subclasses for the specific distributions.
Then when making a network, we only have to specify name if dist. and parameters it takes.
For instance, using a standard gaussian as dist could cool something like

    prior = {'dist': 'normal', 'mu': 0, 'std': 1}
    BNN = BayesianFC([however many nodes], prior=prior)

Or, a gamma dsitribution: (chosen completely random, disregard the specific pdf, focus on the implementation)

    prior = {'dist': 'gamma', 'k': 6.9, 'theta': 3.14}
    BNN = BayesianFC([however many nodes], prior=prior)
"""


class BLinear(BaseModule):
    def __init__(self, in_nodes, out_nodes, prior=None, use_bias=True):
        super().__init__()

        self.device = torch.device("cpu")
        self.use_bias = use_bias
        self.frozen = False

        self.inp = in_nodes
        self.out = out_nodes

        if prior is None:
            prior = {"dist": "gaussian", "params": {"mean": 0., "std": .01}}
        self.prior = prior

        self.distribution = Dists[self.prior["dist"]](in_nodes, out_nodes, **self.prior["params"], device=self.device)
        # self.means = []

    def forward(self, x):
        """
        Pass an input x through layer.
        If layer is 'frozen', compute affine transformation
        else, resample it
        """
        weight, bias = self.distribution()
        # self.means.append(weight.std())

        return F.linear(x, weight, bias)

    def freeze(self):
        """
        Fix the epsilon_parameters, such that the model always returns the same output for a given input
        """
        self.distribution.frozen = True

    def unfreeze(self):
        """
        Unfreezes the epsilon_parameters, such that the model resamples every time.
        """
        self.distribution.frozen = False

    def __str__(self):
        return f"Bayesian Linear (in: {self.inp}, out: {self.out}) layer with {self.prior['dist']}: {self.prior['params']}"

    def kl_loss(self):
        return self.distribution.kl_loss()

