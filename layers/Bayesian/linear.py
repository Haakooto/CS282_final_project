from ..base import BaseModule
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

Still not too sure how to do the priors.
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
    def __init__(self, in_nodes, out_nodes, priors=None, use_bias=True):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if priors is None: priors = {'dist': 'normal'}
        # priors.setdefault("prior_mean", 0)
        # priors.setdefault("prior_var", 1)
        # priors.setdefault("posterior_init")
        # self.priors = priors

        self.W_mu = nn.Parameter(torch.empty(out_nodes, in_nodes, device=self.device))
        self.W_sig = nn.Parameter(torch.empty(out_nodes, in_nodes, device=self.device))

        self.b_mu = nn.Parameter(torch.empty(out_nodes, device=self.device))
        self.b_sig = nn.Parameter(torch.empty(out_nodes, device=self.device))

        self.init_params()

    def init_params(self):
        """"
        initial sampling of parameters
        """
        self.W_mu = self.sample(self.W_mu.size())
        # self.W_mu.data.normal()
        pass

    def sample(self, shape):
        """
        Intended to make it simple to resample parameters when training
        from a preset distribution.
        """
        pass

    def forward(self, x):
        """
        Pass an input x through layer.
        If layer is 'frozen', compute affine transformation
        else, resample it

        """
        if self.W_eps is None:
            W_ = self.sample(self.W_mu.size())
        else:
            pass
            # W_ = self.W_eps * self.W_sig
        W = self.W_mu + W_

        if self.use_bias:
            if self.b_eps is None:
                b_ = self.sample(self.b_mu.size())
            else:
                pass
                # b_ = self.b_eps * self.b_sig
            b = self.b_mu + b_
        else:
            b = None

        return F.linear(x, W, b)

    def freeze(self):
        """
        Fix the epsilon_parameters, such that the model always returns the same output for a given input
        """
        self.W_sig = self.sample(self.W_mu.size())
        if self.use_bias:
            self.b_eps = self.sample(self.b_mu.size())

    def unfreeze(self):
        """
        Unfreezes the epsilon_parameters, such that the model resamples every time.
        """
        self.W_eps = None
        if self.use_bias:
            self.b_eps = None
