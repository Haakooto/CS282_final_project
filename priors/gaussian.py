import torch as torch


class Gaussian(torch.nn.Module):
    """
    Simple Gaussian multivariate prior without correlation.
    """

    def __init__(self, mu=0, sigma=1):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self, shape):
        dist = torch.normal(mean=self.mu, std=self.sigma, size=shape)
        return dist

    def __str__(self):
        return "A gaussian"
