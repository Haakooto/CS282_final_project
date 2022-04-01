import torch
from models.Bayes3FC import Bayesian3FC

BNN = Bayesian3FC(features=20, n1=10, n2=10, n3=10, classes=1)

x = torch.randn(20)
print(x)
y = BNN(x)
print(y)

