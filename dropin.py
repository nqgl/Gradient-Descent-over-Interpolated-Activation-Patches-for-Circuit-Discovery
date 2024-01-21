import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropin(nn.Module):
    def __init__(self, p, randrange=None):
        super().__init__()
        self.p = p
        self.randrange = randrange
    
    def forward(self, x):
        if self.training:
            if self.randrange is None:
                mask = torch.rand_like(x) < self.p
            else:
                mask = torch.randint_like(x, self.randrange) < self.p
            return x.masked_fill(mask, 0)
        else:
            return x