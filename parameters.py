import torch
import torch.nn as nn
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from transformer_lens import HookedTransformer, ActivationCache
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
from torch import Tensor
import einops
import fp_vs_fn
import glob
import os
torch._tensor

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
            (rand + x * mag_prob_adj < p) 
            & (x < max_mag if max_mag is not None else True)
        )
        v = torch.rand_like(x) * v
        # ctx.save_for_backward(mask)
        return x * (~mask) + v * (mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
        # grad at a different value still provides some evidence
            #for what your value should be so I'm keeping these gradients
            # this also allows the use of normal relu without the dying coefficients problem
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

class GradThruRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return F.relu(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

from functools import partial
gradthrurelu = GradThruRelu.apply
gradthrurelu = F.relu
# def gradthrurelu(x, relu=udrelu):
#     return x - (x - relu(x)).detach()


notzero = 0
def gradthruclamp(x, notzero=notzero):
    x = 1 - gradthrurelu(1 - gradthrurelu(x) - notzero * 2) - notzero
    return x
    return NormGrad.apply(btclamp(x))
    # return F.sigmoid(x)




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
    def __init__(
        self, 
        model, 
        resid_coeff_seq :Optional[int] = None,
        p_dropin = 0.2,
        p_dropout = 0.04,
    ) -> None:
        super().__init__()
        rescale = 0.75
        bias = 0.25
        self.modelcfg = model.cfg
        # self.pre = nn.Parameter(
        #     torch.ones(
        #         modelcfg.n_heads,
        #         device=modelcfg.device,
        #         requires_grad=True
        #     ) * rescale
        # )
        self.layers = nn.ParameterList(
            [
                nn.Parameter(
                    torch.rand(
                        (
                            self.modelcfg.n_heads,
                            layer,
                            self.modelcfg.n_heads
                        ),
                        device=self.modelcfg.device,
                        requires_grad=True
                    ) * rescale + bias
                )
                for layer in range(0, self.modelcfg.n_layers)
            ]
        )
        self.post = nn.Parameter(
            torch.ones(
                (
                    self.modelcfg.n_heads,
                    self.modelcfg.n_heads
                ),
                device=self.modelcfg.device,
                requires_grad=True
            ) 
        )
        self.resid_coeffs = nn.Parameter(
            torch.ones(
                (
                    self.modelcfg.n_layers,
                    self.modelcfg.n_heads
                ) if resid_coeff_seq is None else (
                    self.modelcfg.n_layers,
                    resid_coeff_seq,
                    self.modelcfg.n_heads
                ),
                device=self.modelcfg.device,
                requires_grad=True
            )
        )
        self.dropout = nn.Dropout(p_dropout)
        self.dropin = DropInRand(
            p_dropin,
            v=0.5,
            max_mag=0.5,
            mag_prob_adj=0.0
        )


    def drop(self, x):
        return self.dropout(self.dropin(x)) * (1 - self.dropout.p) 

    def interpolate_resid_pres(
        self, 
        rsp_clean :Float[Tensor, "batch pos d_model"], 
        rsp_dirty :Float[Tensor, "batch pos d_model"],
        layer :Optional[int] = 0,
        parallelism :int = 1,
        start_head :Optional[int] = None
    ) -> Float[Tensor, "parallelism*batch pos d_model"]:
        batch_size = rsp_clean.shape[0]
        coefficients = self.c_RS()[layer] if layer is not None else self.c_RS()
        # coefficients = gradthruclamp(self.drop(coefficients))
        # print(start_head, parallelism, rsp_clean.shape, rsp_dirty.shape, coefficients.shape)
        if start_head is not None:
            sl = slice(start_head, parallelism + start_head)
        else:
            sl = slice(None)
        return mylerp(
            parallelize(rsp_clean, parallelism),
            parallelize(rsp_dirty, parallelism),
            batchify(coefficients[sl], batch_size).unsqueeze(-1).unsqueeze(-1)
        )

    # @torch.no_grad()
    # def reset_RS_coeffs(self, new :float = 1.0):
    #     self.resid_coeffs.data.fill_(new)
    
    @torch.no_grad()
    def reset_RS_coeffs(self, new :Optional[float] = 1.0, p :float = 0.5, out=False):
        rand = torch.rand_like(self.resid_coeffs.data)
        self.resid_coeffs.data[
            (rand < p) & (self.resid_coeffs.data < 0)
        ] = new if new is not None else torch.rand_like(self.resid_coeffs.data[rand < p])
        if out:
            rand = torch.rand_like(self.post.data)
            self.post.data[
                (rand < p) & (self.post.data < 0)
            ] = new if new is not None else torch.rand_like(self.post.data[rand < p])


    @torch.no_grad()
    def reset_edges(self, new, p=1.0):
        for layer in self.layers:
            layer.data[
                (torch.rand_like(layer.data) < p)
                & (layer < 0)
            ] = new
        


    @torch.no_grad()
    def clamp_params(self, margin = 0.0):
        self.resid_coeffs.data.clamp_(0 + margin, 1 - margin)
        for layer in self.layers:
            layer.data.clamp_(0 + margin, 1 - margin)
        self.post.data.clamp_(0 + margin, 1 - margin)

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
    
    def c_RS(self):
        return gradthruclamp(self.drop(self.resid_coeffs))
    
    def l1(self, l1_coeff = 0.0, l1_coeff_pre=None, l1_coeff_post=None, crab = 0.01, bias = 0):
        l1_coeff_pre = l1_coeff if l1_coeff_pre is None else l1_coeff_pre
        l1_coeff_post = l1_coeff if l1_coeff_post is None else l1_coeff_post
        def crabs(x):
            """
            crab = 0 makes this behave like a relu
            """
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
            (self.resid_coeffs > 0).sum()
            + sum(
                (layer > 0).sum()
                for layer in self.layers
            )
            + (self.post > 0).sum()
        )
    
    def num_intermediate(self):
        return (
            ((self.resid_coeffs > 0) & (self.resid_coeffs < 1)).sum()
            + ((self.post > 0) & (self.post < 1)).sum()
            + sum(
                ((layer > 0) & (layer < 1)).sum()
                for layer in self.layers
            )
        )


    def print_connections(self, threshold = 0.5):
        # for layer in range(self.modelcfg.n_layers):
        #     print("\nRS -> ", end="")        
        #     for head in range(self.modelcfg.n_heads):
        #         if self.resid_coeffs[layer][head] > threshold:
        #             print(f"{layer}.{head}, ", end="")
        print(f"\n--------\tthresh={threshold}")
        print()
        for layer in range(self.modelcfg.n_layers):
            for head in range(self.modelcfg.n_heads):
                if torch.any(self.layers[layer][head] > threshold):
                    print(f"{layer}.{head} <- ", end="")
                    if self.resid_coeffs[layer][head] > threshold:
                        print(f"RS_0 | <-", end="")

                    for layer2 in range(self.layers[layer].shape[1]):
                        for head2 in range(self.modelcfg.n_heads):
                            if self.layers[layer][head, layer2, head2] > threshold:
                                print(f"{layer2}.{head2}, ", end="")
                    print()

                elif self.resid_coeffs[layer][head] > threshold:
                    print(f"{layer}.{head} <- RS_0")
        print("\n| ", end="")
        for layer in range(self.modelcfg.n_layers):
            for head in range(self.modelcfg.n_heads):
                if self.post[layer, head] > threshold:
                    print(f"{layer}.{head}, ", end="")
        print(f"-> out", end="")
        print()
        print()

    def get_heads(self, threshold = notzero):
        heads = set()
        for layer in range(self.modelcfg.n_layers):
            for head in range(self.modelcfg.n_heads):
                if torch.any(self.layers[layer][head] > threshold):
                    heads.add(f"{layer}.{head}")
                elif self.resid_coeffs[layer][head] > threshold:
                    heads.add(f"{layer}.{head}")
            if self.post[layer, head] > threshold:
                heads.add(f"{layer}.{head}")
        return heads
    
    def print_tp_fp(self, threshold = notzero):
        heads = self.get_heads(threshold)
        fp_vs_fn.fp_tp_edges(self)
        tp, fp = fp_vs_fn.check_list(heads)



    def get_edges(self, threshold = 0.5):
        edges = []
        for layer in range(self.modelcfg.n_layers):
            for head in range(self.modelcfg.n_heads):
                if torch.any(self.layers[layer][head] > threshold):
                    for layer2 in range(self.layers[layer].shape[1]):
                        for head2 in range(self.modelcfg.n_heads):
                            if self.layers[layer][head, layer2, head2] > threshold:
                                edges += [((layer2, head2), (layer, head))]
                if self.resid_coeffs[layer][head] > threshold:
                    edges += [("resid", (layer, head))]
                if self.post[layer, head] > threshold:
                    edges += [((layer, head), "out")]
        return edges
    
    def save(self, path = None, version = None, i = None):
        if version is None:
            version = self.get_latest_version() + 1
                
        if path is None:
            save_dir = f"saved_models/version_{version}"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            path = f"{save_dir}/patcher_{i}.pt"
        torch.save(self, path)

    @classmethod
    def get_latest_version(cls):
        versions = glob.glob("saved_models/version_*")
        versions = [int(v.split("_")[-1]) for v in versions]
        if len(versions) == 0:
            version = -1
        else:
            version = max(versions)
        return version
        
    @classmethod
    def load_by_version(cls, version, model = None):
        iterations = glob.glob(f"saved_models/version_{version}/patcher_*.pt")
        iterations = [int(v.split("_")[-1].split(".")[0]) for v in iterations]
        iteration = max(iterations)
        return cls.load(f"saved_models/version_{version}/patcher_{iteration}.pt", model)
    
    @classmethod
    def load(cls, path, model=None):
        m = torch.load(path)
        if model is not None:
            m.modelcfg = model.cfg
        return m
    
