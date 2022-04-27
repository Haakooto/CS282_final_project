import torch as torch
from torch import nn


class Gaussian(nn.Module):
    """
    Simple Gaussian multivariate prior without correlation.
    """

    def __init__(self, in_features, out_features, mean=0, std=0.001, use_bias=True, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.frozen = False
        self.use_bias = True

        self.prior_mu = mean
        self.prior_sigma = std
        self.posterior_mu_initial = [mean, std]
        self.posterior_rho_initial = [-3, std]

        # register parameters so torch knos which params need to be optimized during backprop
        self.W_mu = nn.Parameter(torch.empty((out_features, in_features), device=self.device, dtype=dtype))
        self.W_rho = nn.Parameter(torch.empty((out_features, in_features), device=self.device, dtype=dtype))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((out_features), device=self.device, dtype=dtype))
            self.bias_rho = nn.Parameter(torch.empty((out_features), device=self.device, dtype=dtype))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Fill parameter tensors with data drawn from distribution (normal)
        """
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def __call__(self):
        if not self.frozen:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return weight.to(torch.double), bias.to(torch.double)

    def __str__(self):
        return "A gaussian"

    def kl_loss(self):
        return 0

