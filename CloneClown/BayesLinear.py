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
    def __init__(self, *, inn, out, loc=0, scale=1, dist="normal", device=None, dtype=None, use_bias=True):
        super().__init__()  # star in start enforcs you to use keyword arguments, and no implicit argument passing

        self.device = device
        self.dtype = dtype
        self.use_bias = use_bias

        self.prior_loc = loc
        self.prior_scale = scale
        self.dist = dist

        # register parameters so pytorch can backprop
        self.weight_loc = nn.Parameter(torch.zeros((out, inn), device=self.device, dtype=self.dtype))
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
        # while the unchanging prior is instancianted and ever remade
        # TODO: Becuase the prior is unchanged, we can precompute the hyu entropy for kl calculation. Will be done later
        if self.dist == "normal":
            self.distribution = torch.distributions.Normal
            self.prior = torch.distributions.Normal(loc=torch.ones_like(self.weight_loc) * self.prior_loc,
                                                  scale=torch.ones_like(self.weight_rho) * self.prior_scale)
        elif self.dist == "vmf":
            self.distribution = VonMisesFisher  # This is a very active debuggng area, I was unable to do this with vmf, and too late to figure out why
            self.prior = HypersphericalUniform(out - 1)  # minus 1 because ¯\_(ツ)_/¯, This is what the authors did, so this is a possble debugging area

        # Fill weight with initial data
        self.weight_loc.data = self.reparameterize(      ).sample()
        self.weight_rho.data = self.reparameterize(loc=-3).sample()

    def reparameterize(self, loc=None, scale=None):
        # default to prior if not provided, though this is only used in the initial reparameterization done at end of init
        if loc is None:
            loc = self.prior_loc
        if scale is None:
            scale = self.prior_scale
        if not isinstance(loc, torch.Tensor):
            loc = torch.ones_like(self.weight_loc) * loc
        if not isinstance(scale, torch.Tensor):
            scale = torch.ones_like(self.weight_rho) * scale

        return self.distribution(loc=loc, scale=scale)

    def forward(self, x):
        weight_scale = torch.log1p(torch.exp(self.weight_rho))  # still do this to ensure positive stds

        self.weight_poste = self.reparameterize(loc=self.weight_loc, scale=weight_scale)  # save distribution as self.weight_pos, so we can calculate kl later

        weight = self.weight_poste.rsample()  # simply sample weights straight from distribution, instead of going complex stuff like before. Life is good

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
        kl = torch.distributions.kl.kl_divergence(self.weight_poste, self.prior)
        """  # ! Using non-bayesian biases
        if self.use_bias:
            kl += torch.distributions.kl.kl_divergence(self.bias_poste, self.bias_prior)
        """
        return kl

