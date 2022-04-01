import torch as torch

class Gaussian(torch.nn.Module):
    """
    Simple Gaussian multivariate prior without correlation. 
    """
    def __init__(self, mu = 0, sigma = 1):
        self.mu = torch.nn.Parameter(torch.tensor(float(mu)))
        self.sigma = torch.nn.Parameter(torch.tensor(float(sigma)))

    def __call__(self, shape):
        dist = torch.normal(mean = self.mu, std = self.sigma, size = shape)
        return dist
