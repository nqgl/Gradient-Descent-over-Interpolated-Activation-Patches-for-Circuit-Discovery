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

def batchify(ptensor, batch_size):
    return einops.repeat(
        ptensor,
        "parallel ... -> (parallel batch) ...",
        batch=batch_size
    )

def mylerp(a, b, w):
    out = a + (b - a) * w
    assert out.shape == a.shape == b.shape
    return out


class GradThruRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return F.relu(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
gradthrurelu = GradThruRelu.apply

def gradthrurelu(x):
    return x - (x - F.relu(x)).detach()


notzero = 0
def gradthruclamp(x):
    return 1 - gradthrurelu(1 - gradthrurelu(x) - notzero)




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
        bias = 0.25
        self.model :HookedTransformer = nonmember_module(model)
        # self.pre = nn.Parameter(
        #     torch.ones(
        #         model.cfg.n_heads,
        #         device=model.cfg.device,
        #         requires_grad=True
        #     ) * rescale
        # )
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
                    ) * rescale + bias
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
            ) * rescale + bias
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
            ) * rescale + bias
        )
        self.drop = nn.Dropout(0.1)



    def interpolate_resid_pres(
        self, 
        rsp_clean :Float[Tensor, "batch pos d_model"], 
        rsp_dirty :Float[Tensor, "batch pos d_model"],
        layer :Optional[int] = 0,
        parallelism :int = 1,
        start_head :Optional[int] = None
    ) -> Float[Tensor, "parallelism*batch pos d_model"]:
        batch_size = rsp_clean.shape[0]
        coefficients = self.resid_coeffs[layer] if layer is not None else self.resid_coeffs
        coefficients = gradthruclamp(self.drop(coefficients))
        print(start_head, parallelism, rsp_clean.shape, rsp_dirty.shape, coefficients.shape)
        if start_head is not None:
            sl = slice(start_head, parallelism + start_head)
        else:
            sl = slice(None)
        return parallelize(rsp_clean, parallelism).lerp(
            parallelize(rsp_dirty, parallelism),
            batchify(coefficients[sl], batch_size).unsqueeze(-1).unsqueeze(-1)
        )



    def print_connections(self, threshold = 0.5):
        # for layer in range(self.model.cfg.n_layers):
        #     print("\nRS -> ", end="")        
        #     for head in range(self.model.cfg.n_heads):
        #         if self.resid_coeffs[layer][head] > threshold:
        #             print(f"{layer}.{head}, ", end="")
        print(f"\n--------\tthresh={threshold}")
        print()
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                if torch.any(self.layers[layer][head] > threshold):
                    print(f"{layer}.{head} <- ", end="")
                    if self.resid_coeffs[layer][head] > threshold:
                        print(f"RS_0 | <-", end="")

                    for layer2 in range(self.layers[layer].shape[1]):
                        for head2 in range(self.model.cfg.n_heads):
                            if self.layers[layer][head, layer2, head2] > threshold:
                                print(f"{layer2}.{head2}, ", end="")
                    print()

                elif self.resid_coeffs[layer][head] > threshold:
                    print(f"{layer}.{head} <- RS_0")
        print("\n| ", end="")
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                if self.post[layer, head] > threshold:
                    print(f"{layer}.{head}, ", end="")
        print(f"-> out", end="")
        print()
        print()

    @torch.no_grad()
    def clamp_params(self):
        self.resid_coeffs.data.clamp_(0, 1)
        for layer in self.layers:
            layer.data.clamp_(0, 1)
        self.post.data.clamp_(0, 1)

    def clamplist(self):
        return gradthruclamplist(self.tolist())

    def tolist(self):
        raise NotImplementedError
        return [
            # self.pre.data,
            *[layer.data for layer in self.layers[1:]],
            self.post.data
        ]

    def c_layers(self):
        return gradthruclamplist([self.drop(layer) for layer in self.layers])
    
    def c_out(self):
        return gradthruclamp(self.drop(self.post))
    
    def l1(self, l1_coeff = 0.0, l1_coeff_pre=None, l1_coeff_post=None, crab = 0.01, bias = 0):
        l1_coeff_pre = l1_coeff if l1_coeff_pre is None else l1_coeff_pre
        l1_coeff_post = l1_coeff if l1_coeff_post is None else l1_coeff_post
        def crabs(x):
            return torch.relu((x + bias) * (1 + crab)) - (x + bias).abs() * crab
        return (
            l1_coeff_pre * crabs(self.resid_coeffs).sum()
            + sum(
                l1_coeff * crabs(layer).sum()
                for layer in self.layers
            )
            + l1_coeff_post * crabs(self.post).sum()
        )
    
    def l0(self):
        return (
            (self.resid_coeffs > notzero).sum()
            + sum(
                (layer > notzero).sum()
                for layer in self.layers
            )
            + (self.post > notzero).sum()
        )