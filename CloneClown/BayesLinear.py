from torch import nn
import torch.nn.functional as F
import torch
from hyperspherical_uniform import HypersphericalUniform
from von_mises_fisher import VonMisesFisher

"""
Bayesian Linear layer. Takes all the parameters for the distribution as keyword arguments
Only the weights are bayesian. The biases are frequentist, which Leif ensured me would be okay
And the model performs fucking great as such, and changing it back will make my life hell

"""
class Linear(nn.Module):
    def __init__(self, *, inn, out, loc=0, scale=1, dist="normal", dist_kwargs={}, device=None, dtype=None, use_bias=True):
        super().__init__()  # star in start enforcs you to use keyword arguments, and no implicit argument passing

        self.device = device
        self.dtype = dtype
        self.use_bias = use_bias

        self.prior_loc = loc
        self.prior_scale = scale
        self.dist = dist
        self.dist_kwargs = dist_kwargs

        self.history = {"mean": [], "std": []}

        # register parameters so pytorch can backprop
        self.weight_loc = nn.Parameter(torch.zeros((out, inn), device=self.device, dtype=self.dtype))
        inn = inn if dist != "vmf" else 1  # the vmf only taked 1 in-feature for the scale, though shapes work out later to be same as before. Not sure why
        self.weight_rho = nn.Parameter(torch.zeros((out, inn), device=self.device, dtype=self.dtype))

        """ # ! Using non-bayesian biases
        if self.use_bias:
            self.bias_loc = nn.Parameter(torch.zeros(out, device=self.device, dtype=self.dtype))
            self.bias_rho = nn.Parameter(torch.zeros(out, device=self.device, dtype=self.dtype))
        else:
            self.register_parmeter("bias_loc", None)
            self.register_parmeter("bias_rho", None)
        """ # ! Swap with the following if bayesian biases are wanted. Then self.distribution will have to be changed dough, so please heavily consider is this is actually nessissary, or only done for shits and giggles and you think it's fun fucking with my night-sleep quality. Tbf, is shouldn't be too bad, but its still annoying and I have now gone back and forth for almost 2 hours about how the code should be structued in general, and that function has been very centrial in exactly why it took this long, and I'm tired and only writing this long ass-comment to further procrastinating writing the final version that will magically work and win us the nobel price in economics in 5 years. Yes, economics. Because there is no nobel price for stiring a pile of shit until it accurately tells you the age of your dog, which lets be honest, is all we are doing. No, we have to think better. When this fucktion works we will be billionares and solve half of all world problems with out fantastic super-general bayesian neural network! And should my prediction turn out to not be right I think we're all better off taking Vincent up on his offer.
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out, device=self.device, dtype=self.dtype))
        else:
            self.register_parmeter("bias", None)

        # Setting distributions. The posterior is the untriggered distribution,
        # while the unchanging prior is instancianted and never remade
        # TODO: Becuase the prior is unchanged, we can precompute the hyu entropy for kl calculation. Will be done later.
        # ? Thanks, this saved me 2.6% of training time.
        if self.dist == "normal":
            self.distribution = torch.distributions.Normal
            self.weight_prior = self.distribution(loc=torch.ones_like(self.weight_loc) * self.prior_loc,
                                                  scale=torch.ones_like(self.weight_rho) * self.prior_scale,
                                                  **self.dist_kwargs,
                                                  )  # initialize once, then only use to calc kl after
        elif self.dist == "vmf":
            self.distribution = VonMisesFisher  # This is a very active debuggng area, I was unable to do this with vmf, and too late to figure out why
            self.weight_prior = HypersphericalUniform(out - 1,   # minus 1 because ¯\_(ツ)_/¯, This is what the authors did, so this is a possble debugging area
                                                      device=self.device,
                                                      )
        else:
            raise NotImplementedError

        # Fill weight with initial data
        self.weight_loc.data = self.distribution(loc=torch.ones_like(self.weight_loc) * self.prior_loc,
                                                 scale=torch.ones_like(self.weight_rho) * self.prior_scale,
                                                 **self.dist_kwargs,
                                                 ).sample()
        self.weight_rho.data = self.distribution(loc=torch.ones_like(self.weight_loc) * -3,  # -3 for small initial variance
                                                 scale=torch.ones_like(self.weight_rho) * self.prior_scale,
                                                 **self.dist_kwargs,
                                                 ).sample()
        # print(self.weight_loc.size())
        # self.weight_loc.data = self.weight_prior.sample()
        # self.weight_rho.data = self.weight_prior.sample()
        # print(self.weight_loc.size())
        if self.dist == "vmf":  # The sampling above adds too many dimensions in last axis of weight_rho
            self.weight_rho.data = self.weight_rho.data.mean(-1, keepdim=True)  # I quite arbitrarily average this to make shapes work. Its just initialisation, so what could be wrong?

    def forward(self, x):
        weight_scale = torch.log1p(torch.exp(self.weight_rho))  # still do this to ensure positive stds

        self.weight_posterior = self.distribution(loc=self.weight_loc, scale=weight_scale, **self.dist_kwargs)  # save distribution as self.weight_pos, so we can calculate kl later

        weight = self.weight_posterior.rsample()  # simply sample weights straight from distribution, instead of going complex stuff like before. Life is good
        self.history["mean"].append(weight.mean())
        self.history["std"].append(weight.std())
        
        """  # ! Using non-bayesian biases
        if self.use_bias:
            bias_scale = torch.log1p(torch.exp(self.bias_rho))

            self.bias_poste = self.distribution(loc=self.bias_loc, scale=bias_scale)
            self.bias_prior = self.distribution(prior=True)

            bias = self.bias_poste.rsample()
        else:
            bias = None

        return F.linear(x, weight, bias)
        """
        return F.linear(x, weight, self.bias)

    def kl_div(self):  # GUYS! we dont need to use out triangular wheel when torch already has a circular one
        kl = torch.distributions.kl.kl_divergence(self.weight_posterior, self.weight_prior)
        """  # ! Using non-bayesian biases
        if self.use_bias:
            kl += torch.distributions.kl.kl_divergence(self.bias_poste, self.bias_prior)
        """
        return kl

