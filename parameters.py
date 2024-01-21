import torch
import torch.nn as nn
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from transformer_lens import HookedTransformer, ActivationCache
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
from torch import Tensor
import einops


def parallelize(btensor, parallelism):
    return einops.repeat(
        btensor,
        "batch ... -> (parallelism batch) ...",
        parallelism=parallelism
    )

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


class nonmember_module():
    def __init__(self, m):
        self.m = m

    def __getattribute__(self, __name: str):
        if __name == '__class__':
            return object.__getattribute__(self, '__class__')
        m = object.__getattribute__(self, 'm')
        return getattr(m, __name)

# m = nonmember_module(GradThruRelu)
# m.apply

    

class InterpolatedPathPatch(nn.Module):
    def __init__(self, model, resid_coeff_seq :Optional[int] = None) -> None:
        super().__init__()
        rescale = 0.5
        self.model :HookedTransformer = nonmember_module(model)
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
        self.resid_coeffs = nn.Parameter(
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



    def interpolate_resid_pres(
        self, 
        rsp_clean :Float[Tensor, "batch pos d_model"], 
        rsp_dirty :Float[Tensor, "batch pos d_model"],
        layer :Optional[int] = 0,
        parallelism :int = 1,
        head :Optional[int] = None,
    ) -> Float[Tensor, "(parallelism batch) pos d_model"]:
        '''
        Interpolate between the clean and dirty residuals
        '''
        coefficients = self.resid_coeffs[layer] if layer is not None else self.resid_coeffs
        if head is not None:
            sl = slice(parallelism * head, parallelism * (head + 1))
        else:
            sl = slice(None)
        return parallelize(rsp_clean, parallelism).lerp(
            parallelize(rsp_dirty, parallelism),
            coefficients[sl]
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