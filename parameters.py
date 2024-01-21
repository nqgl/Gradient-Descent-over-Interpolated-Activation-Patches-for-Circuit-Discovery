import torch
import torch.nn as nn
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

import torch.nn.functional as F


class GradThruRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return F.relu(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
gradthrurelu = GradThruRelu.apply
def gradthruclamp(x):
    return 1 - gradthrurelu(1 - gradthrurelu(x))



def gradthruclamplist(l):
    return [gradthruclamp(x) for x in l]




class InterpolatedPathPatch(nn.Module):
    def __init__(self, model, resid_coeff_seq :Optional[int] = None) -> None:
        super().__init__()
        rescale = 0.5
        self.model = model
        self.pre = nn.Parameter(
            torch.ones(
                model.cfg.n_heads,
                device=model.cfg.device,
                requires_grad=True
            ) * rescale
        )
        self.layers = nn.ParameterList(
            [
                nn.Parameter(
                    torch.rand(
                        (
                            model.cfg.n_heads,
                            layer,
                            model.cfg.n_heads
                        ),
                        device=model.cfg.device,
                        requires_grad=True
                    ) * rescale
                )
                for layer in range(0, model.cfg.n_layers)
            ]
        )
        self.post = nn.Parameter(
            torch.rand(
                (
                    model.cfg.n_heads,
                    model.cfg.n_heads
                ),
                device=model.cfg.device,
                requires_grad=True
            ) * rescale
        )
        self.resid_coeffs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.rand(
                        (
                            model.cfg.n_layers,
                            model.cfg.n_heads
                        ) if resid_coeff_seq is None else (
                            model.cfg.n_layers,
                            resid_coeff_seq,
                            model.cfg.n_heads
                        ),
                        device=model.cfg.device,
                        requires_grad=True
                    ) * rescale
                )
                for layer in range(0, model.cfg.n_layers)
            ]
        )



    def print_connections(self, threshold = 0.5):
        print("\n\nRS -> ", end="")
        for head in range(self.model.cfg.n_heads):
            if self.pre[head] > threshold:
                print(f"0.{head}, ", end="")
        print()
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                if torch.any(self.layers[layer][head] > threshold):
                    print(f"{layer}.{head} <- ", end="")
                    for layer2 in range(self.layers[layer].shape[1]):
                        for head2 in range(self.model.cfg.n_heads):
                            if self.layers[layer][head, layer2, head2] > threshold:
                                print(f"{layer2}.{head2}, ", end="")
                    print()
        print(f"\nout <- ", end="")
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                if self.post[layer, head] > threshold:
                    print(f"{layer}.{head}, ", end="")
        print()

    def clamplist(self):
        return gradthruclamplist(self.tolist())

    def tolist(self):
        return [
            self.pre.data,
            *[layer.data for layer in self.layers[1:]],
            self.post.data
        ]