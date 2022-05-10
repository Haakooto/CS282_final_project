from torch import nn
import torch.nn.functional as F
import torch
from hyperspherical_uniform import HypersphericalUniform
from von_mises_fisher import VonMisesFisher
from power_spherical import PowerSpherical

"""
Bayesian Linear layer. Takes all the parameters for the distribution as keyword arguments
Only the weights are bayesian. The biases are frequentist, which Leif ensured me would be okay
And the model performs fucking great as such, and changing it back will make my life hell

"""
class Linear(nn.Module):
    def __init__(self, *, inn, out, loc=0, scale=1, dist="normal", dist_kwargs={}, device=None, dtype=None, use_bias=True, record=False):
        super().__init__()  # star in start enforces you to use keyword arguments, and no implicit argument passing

        self.device = device
        self.dtype = dtype
        self.use_bias = use_bias
        self.record = record

        self.prior_loc = loc
        self.prior_scale = scale
        self.dist = dist
        self.dist_kwargs = dist_kwargs

        self.history = {"mean": [], "std": [], "all": [], "first": []}

        # register parameters so pytorch can backprop
        self.weight_loc = nn.Parameter(torch.zeros((out, inn), device=self.device, dtype=self.dtype))
        inn = inn if dist not in ("vmf", "power") else 1  # the vmf only taked 1 in-feature for the scale, though shapes work out later to be same as before. Not sure why
        self.weight_rho = nn.Parameter(torch.zeros((out, inn), device=self.device, dtype=self.dtype))

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out, device=self.device, dtype=self.dtype))
        else:
            self.register_parmeter("bias", None)

        # Setting distributions. The posterior is the untriggered distribution,
        # while the unchanging prior is instancianted and never remade
        if self.dist == "normal":
            self.distribution = torch.distributions.Normal
            self.prior = self.distribution(loc=torch.ones_like(self.weight_loc) * self.prior_loc,
                                           scale=torch.ones_like(self.weight_rho) * self.prior_scale,
                                           **self.dist_kwargs,
                                           )  # initialize once, then only use to calc kl after
        elif self.dist == "vmf":
            self.distribution = VonMisesFisher
            self.prior = HypersphericalUniform(out - 1,   # minus 1 because ¯\_(ツ)_/¯, This is what the authors did
                                                      device=self.device,
                                                      )
        elif self.dist == "power":
            self.distribution = PowerSpherical
            self.prior = HypersphericalUniform(out - 1,   # minus 1 because ¯\_(ツ)_/¯, This is what the authors did
                                                      device=self.device,
                                                      )
        else:
            raise NotImplementedError

        # Fill weight with initial data
        self.weight_loc.data = self.distribution(loc=torch.rand(self.weight_loc.size(), dtype=torch.double) * self.prior_loc, #self.prior_loc
                                                 scale=torch.ones_like(self.weight_rho) * self.prior_scale, #self.prior_scale
                                                 **self.dist_kwargs,
                                                 ).sample()
        self.weight_rho.data = self.distribution(loc=torch.ones_like(self.weight_loc) * -3,  # -3 for small initial variance
                                                 scale=torch.ones_like(self.weight_rho) * self.prior_scale, #self.prior_scale
                                                 **self.dist_kwargs,
                                                 ).sample()

        if self.dist in ("vmf", "power"):  # The sampling above adds too many dimensions in last axis of weight_rho
            self.weight_rho.data = self.weight_rho.data.mean(-1, keepdim=True)  # I quite arbitrarily average this to make shapes work. Its just initialisation, so what could be wrong?

    def forward(self, x):
        weight_scale = F.softplus(self.weight_rho)  # ensure positive scale

        self.posterior = self.distribution(loc=self.weight_loc, scale=weight_scale, **self.dist_kwargs)  # save distribution as self.weight_pos, so we can calculate kl later

        weight = self.posterior.rsample() # simply sample weights straight from distribution, instead of going complex stuff like before. Life is good
        if self.record:
            self.history["mean"].append(weight.mean())
            self.history["std"].append(weight.std())
            self.history["all"].append(weight)
            self.history["first"].append(weight[0, 0])

        return F.linear(x, weight, self.bias)

    def kl_div(self):
        return torch.distributions.kl.kl_divergence(self.posterior, self.prior)

