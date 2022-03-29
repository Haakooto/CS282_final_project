import torch
from models.bayesian.Bayes3fc import Bayesian3FC

BNN = Bayesian3FC(20, 10, 10, 10)

print(list(BNN.modules()))

x = torch.randn(20)
print(x)
y = BNN(x)
print(y)

