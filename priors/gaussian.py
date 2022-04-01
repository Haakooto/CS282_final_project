import torch as torch

class gaussian:
    """
    Simple layer skeleton. Can be inherited for making full models.
    Should also add calculation of KL-loss, for bayesian stuff. See papers
    """
    def __init__(self, mu = 0, sigma = 1):
        self.mu = mu
        self.sigma = sigma


    def __call__(self, shape):
        dist = torch.normal(mean = self.mu, std = self.sigma, size = shape)
        return dist
