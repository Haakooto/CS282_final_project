import torch
from BayesModel import BFC, FC

net = BFC(features=5, classes=3, hiddens=[20, 20], prior={"dist": "vmf", "loc": 0, "scale": 1, "dist_kwargs": {"k": 1}})
# net = BFC(features=5, classes=3, hiddens=[20, 20], prior={
#           "dist": "normal", "loc": 0, "scale": 0.1})

x = torch.randn(5).to(torch.double)
print(x)
y = net(x)
print(y)

# for i in [0, 2, 4]:
#     L = net.layers[i]
#     prior = L.weight_prior
#     # post = torch.distributions.Normal(loc=prior.loc, scale=prior.scale)
#     post = L.weight_posterior
#     kl_div = torch.distributions.kl.kl_divergence
#     print(kl_div(post, prior).mean())
