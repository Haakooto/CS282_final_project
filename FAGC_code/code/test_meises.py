from von_mises_fisher import VonMisesFisher as VMF
from hyperspherical_uniform import HypersphericalUniform as HSU
from power_spherical import PowerSpherical as POWWAH
import torch

dim = 8
loc = torch.tensor([0.] * (dim - 1) + [1.])
scale = torch.tensor(10.)

dist1 = POWWAH(loc, scale)
dist2 = HSU(dim)
x = dist1.sample((100000,))

result = torch.isclose(
    (dist1.log_prob(x) - dist2.log_prob(x)).mean(),
    torch.distributions.kl_divergence(dist1, dist2),
    atol=1e-2,
).all()

print(result)