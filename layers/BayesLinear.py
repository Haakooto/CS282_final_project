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


class Linear(BaseModule):
    def __init__(self, in_nodes, out_nodes, prior=None, use_bias=True):
        super().__init__()

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.use_bias = use_bias
        self.frozen = False

        if prior is None:
            # prior = {"dist": "gaussian", "params": {"w_mu": 0, "w_sigma": 1, "b_mu": 1, "b_sigma":}}
            prior = {"dist": "gaussian", "params": {"mu": 0, "sigma": 1}}
        self.prior = prior

        # print(Dists[self.prior["dist"]]())
        # exit()

        self.distribution = Dists[self.prior["dist"]](**self.prior["params"], device=self.device)

        self.weight = nn.Parameter(torch.empty(in_nodes, out_nodes, device=self.device, dtype=float))
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_nodes, device=self.device, dtype=float))
        else:
            self.register_parameter("bias", None)

        self.sample()

    def sample(self):
        """
        Intended to make it simple to resample parameters when training
        from a preset distribution.
        """
        self.weight.data = self.distribution(self.weight.size())
        if self.use_bias:
            self.bias.data = self.distribution(self.bias.size())

    def forward(self, x):
        """
        Pass an input x through layer.
        If layer is 'frozen', compute affine transformation
        else, resample it
        """
        if not self.frozen:
            self.sample()

        # return F.linear(x, self.weight, self.bias)
        return x @ self.weight + self.bias

    def freeze(self):
        """
        Fix the epsilon_parameters, such that the model always returns the same output for a given input
        """
        self.sample()
        self.frozen = True

    def unfreeze(self):
        """
        Unfreezes the epsilon_parameters, such that the model resamples every time.
        """
        self.frozen = False
