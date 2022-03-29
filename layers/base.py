import torch
from torch import nn
import torch.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x