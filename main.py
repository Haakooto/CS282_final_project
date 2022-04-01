import torch
from models.Bayes3FC import Bayesian3FC
from priors.gaussian import gaussian



x = torch.randn(20)
print(x)
g = gaussian()
print(g(x.shape))

# BNN = Bayesian3FC(features=20, n1=10, n2=10, n3=10, classes=1)
# y = BNN(x)
# print(y)

