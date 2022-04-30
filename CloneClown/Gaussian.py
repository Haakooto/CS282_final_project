import torch
import torch.distributions as dist
from torch import nn


class Gauss(nn.Module):
    def __init__(self, inn, out, *, mean, std, bias, kl_adder):
        super().__init__()
        # should not be any surprises here compared with our earlier code
        self.use_bias = bias
        self.kl_adder = kl_adder
        self.unfrozen = True

        device = torch.device("cpu")
        dtype = torch.double

        self.prior_mean = mean
        self.prior_std = std
        self.post_mean = [mean, std]
        self.post_rho = [-3, std]

        self.training_data = {"wm": [], "ws": [], "bm": [], "bs": []}

        self.weight_mean = nn.Parameter(torch.zeros((out, inn), device=device, dtype=dtype))
        self.weight_rho  = nn.Parameter(torch.zeros((out, inn), device=device, dtype=dtype))
        if self.use_bias:
            self.bias_mean = nn.Parameter(torch.zeros(out, device=device, dtype=dtype))
            self.bias_rho  = nn.Parameter(torch.zeros(out, device=device, dtype=dtype))
        else:
            self.register_parmeter("bias_mean", None)
            self.register_parmeter("bias_rho", None)

        self.prior = dist.Normal(0, 1)
        self.reset()

    def reset(self):
        # initi weights with data
        self.weight_mean.data.normal_(*self.post_mean)
        self.weight_rho.data.normal_(*self.post_rho)
        if self.use_bias:
            self.bias_mean.data.zero_()
            self.bias_rho.data.zero_()

    def __call__(self):
        # resample weights from distribution, just as before
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mean + weight_eps * weight_std * self.unfrozen

        self.kl_loss(weight, self.weight_mean, self.weight_rho)
        # self.training_data["wm"].append(weight.mean())
        # self.training_data["ws"].append(weight.std())

        if self.use_bias:
            bias_std = torch.log1p(torch.exp(self.bias_rho))
            bias_eps = torch.randn_like(bias_std)
            bias = self.bias_mean + bias_eps * bias_std * self.unfrozen

            self.kl_loss(bias, self.bias_mean, self.bias_rho)
            # self.training_data["bm"].append(bias.mean())
            # self.training_data["bs"].append(bias.std())
        else:
            bias = None

        return weight, bias

    def kl_loss(self, z, mu, rho):
        log_prior = self.prior.log_prob(z)
        sigma = torch.log1p(torch.exp(rho))
        log_pq = dist.Normal(mu, sigma).log_prob(z)
        self.kl_adder( (log_pq - log_prior).sum() )


# class Uniform(nn.Module):
#     # dont @ me
#     def __init__(self, inn, out, *, min, max, bias, kl_adder):

#         self.use_bias = bias
#         self.kl_adder = kl_adder
#         self.unfrozen = True

#         device = torch.device("cpu")
#         dtype = torch.double

#         self.prior_min = min
#         self.prior_max = max
#         self.post_mean = [min, max]
#         self.post_spread = [0, 1]

#         self.weight_mean = nn.Parameter(torch.zeros(out, inn), device=device, dtype=dtype)
#         self.weight_rho = nn.Parameter(torch.zeros(out, inn), device=device, dtype=dtype)
#         if self.use_bias:
#             self.bias_mean = nn.Parameter(torch.zeros(out), device=device, dtype=dtype)
#             self.bias_rho = nn.Parameter(torch.zeros(out), device=device, dtype=dtype)
#         else:
#             self.register_parmeter("bias_mean", None)
#             self.register_parmeter("bias_rho", None)

#         self.reset()

#     def reset(self):
#         self.weight_mean.data.uniform_(*self.post_mean)
#         self.weight_rho.data.normal_(*self.post_spread)
#         if self.use_bias:
#             self.bias_mean.data.zero_()
#             self.bias_rho.data.zero_()

#     def __call__(self):
#         weight_spread = torch.log1p(torch.exp(self.weight_rho))
#         weight_eps = torch.rand_like(weight_spread)
#         weight = self.weight_mean + weight_eps * weight_spread * self.unfrozen

#         self.kl_loss(weight, self.weight_mean, self.weight_rho)

#         if self.use_bias:
#             bias_spread = torch.log1p(torch.exp(self.bias_rho))
#             bias_eps = torch.rand_like(bias_spread)
#             bias = self.bias_mean + bias_eps * bias_spread * self.unfrozen

#             self.kl_loss(bias, self.bias_mean, self.bias_rho)
#         else:
#             bias = None

#         return weight, bias

#     def kl_loss(self, z, mu, rho):
#         log_prior = dist.Uniform(self.prior_min, self.prior_max).log_prob(z)
#         sigma =
