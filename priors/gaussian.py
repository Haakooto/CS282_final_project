import torch as torch
from torch import nn
from torch import distributions as dist


class Gaussian(nn.Module):
    """
    Simple Gaussian multivariate prior without correlation.
    """

    def __init__(self, in_features, out_features, parent, mean=0, std=0.001, use_bias=True, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.frozen = False
        self.use_bias = use_bias
        self.parent = parent

        self.prior_mu = mean
        self.prior_sigma = std
        self.posterior_mu_initial = [mean, std]
        self.posterior_rho_initial = [-3, std]

        # register parameters so torch knos which params need to be optimized during backprop
        self.W_mu = nn.Parameter(torch.empty((out_features, in_features), device=self.device, dtype=dtype))
        self.W_rho = nn.Parameter(torch.empty((out_features, in_features), device=self.device, dtype=dtype))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.zeros((out_features), device=self.device, dtype=dtype))
            self.bias_rho = nn.Parameter(torch.zeros((out_features), device=self.device, dtype=dtype))
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

        # if self.use_bias:
        #     self.bias_mu.data._zeros()
        #     self.bias_rho.data._zeros()

    def __call__(self):
        if not self.frozen:
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            W_eps = torch.randn_like(self.W_sigma)
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_eps = torch.randn_like(self.bias_sigma)
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        self.kl_loss(weight, self.W_mu, self.W_rho)
        if self.use_bias:
            self.kl_loss(bias, self.bias_mu, self.bias_rho)

        return weight.to(torch.double), bias.to(torch.double)

    def __str__(self):
        return "A gaussian"

    def kl_loss(self, z, mu, p):
        log_prior = dist.Normal(0, 1).log_prob(z)
        sigma = torch.log1p(torch.exp(p))
        log_pq = dist.Normal(mu, sigma).log_prob(z)
        self.parent.total_kl_div += (log_pq - log_prior).sum() / 429  # 429 is the number of batches. dont @ me

        # kl = kl_normal(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        # if self.use_bias:
        #     kl += kl_normal(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        # return kl


def kl_normal(mu_p, sig_p, mu_q, sig_q):
    return 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()

