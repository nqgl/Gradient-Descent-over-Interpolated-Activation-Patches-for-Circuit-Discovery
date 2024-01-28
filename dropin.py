import torch
import torch.nn as nn
import torch.nn.functional as F

class DropIn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, v):
        mask = torch.rand_like(x) > p
        ctx.save_for_backward(mask)
        return x * mask + v * (~mask)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
        mask, = ctx.saved_tensors
        return grad_output * mask, None, None

class dropInRand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, v=1., max_mag = 1.0, mag_prob_adj = 0.0):
        rand = torch.rand_like(x)
        mask = (
            (rand + F.relu(x) * mag_prob_adj < p) 
            & (x < max_mag if max_mag is not None else True)
        )
        v = torch.rand_like(x) * v
        # ctx.save_for_backward(mask)
        return x * (~mask) + v * (mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
        # I'm thinking that grad at a different value still provides some evidence
            # for what your value should be so I'm keeping these gradients
            # this also allows the use of normal relu without as much of the dying coefficients problem
            # which seems good
        mask, = ctx.saved_tensors
        return grad_output * mask, None, None

class DropInRand(nn.Module):
    def __init__(self, p, v=1., max_mag = 1.0, mag_prob_adj = 0.0):
        super().__init__()
        self.p=p
        self.v=v
        self.max_mag=max_mag
        self.mag_prob_adj=mag_prob_adj

    
    def forward(self, x):
        return dropInRand.apply(x, self.p, self.v, self.max_mag, self.mag_prob_adj)


# class Dropin(nn.Module):
#     def __init__(self, p, randrange=None):
#         super().__init__()
#         self.p = p
#         self.randrange = randrange
    
#     def forward(self, x):
#         if self.training:
#             if self.randrange is None:
#                 mask = torch.rand_like(x) < self.p
#             else:
#                 mask = torch.randint_like(x, self.randrange) < self.p
#             return x.masked_fill(mask, 0)
#         else:
#             return x