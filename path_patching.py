#%%
import torch

from parameters import InterpolatedPathPatch
t = torch
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from ioi_dataset import IOIDataset
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from functools import partial
import einops
import torch.nn.functional as F

# %%
from parameters import InterpolatedPathPatch, parallelize, mylerp
# %%

from load_run import model, get_next_data




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

from nqgl.mlutils.time_gpu import TimedFunc


from nqgl.mlutils.norepr import fastpartial as partial





def extract_tensor_hook(
    heads_output :Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    heads: List,
    head: Optional[int],
    parallelism = 1
    # save_list : List[Dict[int, Optional[Tensor]]],
):
    if parallelism == 1:
        if head is None:
            heads.append(heads_output)
        else:
            heads.append(heads_output[:, :, head, :].unsqueeze(-2))
    else:
        batch_size = heads_output.shape[0]
        segment_size = batch_size // parallelism
        # assert batch_size % parallelism == 0
        outs = []
        assert head is not None
        for i in range(parallelism):
            heads.append(heads_output[i * segment_size : (i + 1) * segment_size, :, head + i, :].unsqueeze(-2))
    return heads_output


def head_interpolate_hook(
    heads_output :Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    clean_cache: ActivationCache,
    patched_heads :Float[Tensor, "layer batch pos head_index d_head"],
    # save_list : List[Dict[int, Optional[Tensor]]],
    patch_coefficients : Float[Tensor, "run_layers head_index"],
    coeffs : InterpolatedPathPatch,
    parallelism = 1
    # ignore_positions : Optional[Bool[Tensor, "layer pos head_index"]] = None,
) -> Float[Tensor, "batch pos head_index d_head"]:
    # print("called hook", hook.name, hook.layer())
    # print(heads_output.shape, patched_heads[hook.layer()].shape, patch_coefficients[hook.layer()].shape)
    # out = heads_output.lerp(
    #     patched_heads[hook.layer()], 
    #     patch_coefficients[hook.layer()].unsqueeze(-1)
    # )
    if parallelism == 1:
        out = mylerp(
            clean_cache[hook.name],
            patched_heads[hook.layer()], 
            patch_coefficients[hook.layer()].unsqueeze(-1)
        )
    else:
        batch_size = heads_output.shape[0]

        segment_size = batch_size // parallelism
        # print("batch_size", batch_size, "segment_size", segment_size)
        assert batch_size % parallelism == 0
        outs = []
        for i in range(parallelism):
            out = mylerp(
                clean_cache[hook.name],
                patched_heads[hook.layer()], 
                patch_coefficients[i][hook.layer()].unsqueeze(-1)
            )
            outs.append(out)
        out = torch.cat(outs, dim=0)
        # print("outs out", out.shape)
    return out

def interpolated_path_patch(
    model: HookedTransformer,
    # patching_metric: Callable,
    # src_nodes : List[Tuple[int, int]],
    # dest_nodes : List[Tuple[int, int]],
    # new_dataset: IOIDataset = abc_dataset,
    # orig_dataset: IOIDataset = ioi_dataset,
    coeffs : InterpolatedPathPatch,
    # patch_coefficients : Optional[List[Float[Tensor, "... head_index"]]] = None,
    new_cache: Optional[ActivationCache],
    orig_cache: Optional[ActivationCache],
    parallelism = 1
) -> Float[Tensor, "layer head"]:

    #step 2


    patch_coefficients = coeffs.c_layers()
    patched_heads_values = []
    for layer in range(0, model.cfg.n_layers):
        patched_heads_in_layer = []
        # print("\nlayer", layer, end="->")
        for head_i in range(model.cfg.n_heads // parallelism):  
            head = head_i * parallelism
            # print(head, end=", ")


            if layer != 0:
                if parallelism == 1:
                    head_coeffs = patch_coefficients[layer][head]
                else:
                    head_coeffs = patch_coefficients[layer][head : head + parallelism]
            else:
                head_coeffs = None
            hook_fn = partial(
                head_interpolate_hook,
                clean_cache=orig_cache,
                coeffs = coeffs,
                patched_heads = patched_heads_values,
                # ignore_positions = None,
                patch_coefficients = head_coeffs,
                parallelism = parallelism
            )

            extract_hook_fn = partial(
                extract_tensor_hook,
                heads = patched_heads_in_layer,
                head = head,
                parallelism = parallelism
            )
            logits = model.run_with_hooks(
                coeffs.interpolate_resid_pres(
                    orig_cache["resid_pre", 0],
                    new_cache["resid_pre", 0],
                    layer=layer,
                    parallelism=parallelism,
                    start_head=head if parallelism != model.cfg.n_heads else None
                ),
                start_at_layer=0,
                stop_at_layer=layer + 1,
                fwd_hooks=[
                    (
                        lambda name: name.endswith("z") and not f".{layer}." in name,
                        hook_fn
                    ), (
                        lambda name: name.endswith("z") and f".{layer}." in name,
                        extract_hook_fn
                    )
                ]
            )
            # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
            # print("done cuda time")
            # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
            # print("done cputime")
            # print()
        zcat = torch.cat(patched_heads_in_layer, dim=-2)
        patched_heads_values += [zcat]
    last_patch_coeffs = coeffs.c_out()
    hook_fn = partial(
        head_interpolate_hook,
        clean_cache=orig_cache,
        coeffs = coeffs,
        patched_heads = patched_heads_values,
        # ignore_positions = None,
        patch_coefficients = last_patch_coeffs,
        parallelism = 1
    )

    extract_hook_fn = partial(
        extract_tensor_hook,
        heads = patched_heads_in_layer,
        head = head,
        parallelism = 1
    )

    logits = model.run_with_hooks(
        orig_cache["resid_pre", 0],
        start_at_layer=0,
        fwd_hooks=[
            (
                lambda name: name.endswith("z"),
                hook_fn
            )
        ]
    )

    return patched_heads_values, patch_coefficients, logits


# patch_coefficients = [patcher.pre] + [*patcher.layers[1:]] + [patcher.post]
# optim = torch.optim.Adam(patcher.parameters(), lr=0.1, weight_decay=0, betas=(0.9, 0.98))

# patch_coefficients, roots = get_per_layer_patch_coefficients(model)
# optim = torch.optim.Adam(roots, lr=0.1, wight_decay=0.01)
# optim = torch.optim.Adam(roots, lr=0.1, weight_decay=0, betas=(0.9, 0.94))
# clamped_coefficients = patcher.clamplist()
# patch_coefficients = [patcher.pre] + [*patcher.layers[1:]] + [patcher.post]
# clamped_coefficients = gradthruclamplist(patch_coefficients)
import glob
import os
versions = glob.glob("saved_models/version_*")
versions = [int(v.split("_")[-1]) for v in versions]
if len(versions) == 0:
    version = 0
else:
    version = max(versions) + 1

LOAD=False
if LOAD:
    version = InterpolatedPathPatch.get_latest_version()
    patcher = InterpolatedPathPatch.load_by_version(version, model)
else:
    patcher = InterpolatedPathPatch(model, p_dropin=0.2, p_dropout=0.05)
optim = torch.optim.SGD(patcher.parameters(), lr = 0.001, momentum=0.8, nesterov=True)
# optim = torch.optim.Adam(patcher.parameters(), lr=0.001, betas=(0.9, 0.99))
N = 12
l0_history = [1.]
import tqdm
# interpolated_path_patch = TimedFunc(interpolated_path_patch, print_on_call=True)
for i in tqdm.tqdm(range(10000)):
    print("\n\nstep ", i)
    optim.zero_grad()
    print("getting data")
    ioi_cache, abc_cache, ioi_metric, kl_metric = get_next_data(N)
    print("getting path patch")
    # with torch.autograd.profiler.profile(
        # with_stack=True, with_modules=True, profile_memory=True) as prof:


    values, coefficients, logits = interpolated_path_patch(
        model, 
        coeffs=patcher,
        new_cache=abc_cache,
        orig_cache=ioi_cache,
        parallelism=12,
    )
    # prof.export_chrome_trace(f"./traces/traceall{i}.json")
    # patch_coefficients = patcher.tolist()

    kl_div = kl_metric(logits)
    loss_l1 = patcher.l1(0.001, 0.001, 0.001, crab = 0.) * 2.5
    loss_logits = ioi_metric(logits)
    l0 = patcher.l0().item()
    l0_history.append(l0)
    # loss = kl_div * 0.02 + loss_logits
    loss = loss_logits
    loss = loss + loss_l1 * (1 + min(l0_history[-6:]) / max(15000 - i * 500, 200)) # if i > 20 else loss_logits + loss_l1
    if i % 100 < 10 or i % 10 == 0:
        if l0 < 400:
            patcher.print_connections(threshold=0)
            patcher.print_connections()
        

        print("\nlogit diff:", loss_logits.item())
        print("nonzero coeffs:", l0)
        print("intermeidate values:", patcher.num_intermediate())
    print("kl_score", kl_div.item())
    print("L1:", loss_l1.item())
    loss.backward()
    optim.step()
    patcher.print_tp_fp(threshold=0.5)
    if i < 2000 and i % 100 == 0:
        patcher.clamp_params(0.01)
    if i < 2000 and i % 200 == 5:
        patcher.clamp_params(0.01)
        patcher.reset_RS_coeffs(0.9, p=0.5, out=True)
        patcher.reset_edges(0.1, p=0.25)
    # if i == 10:
    #     patcher.clamp_params()
    # clamped_coefficients = patcher.clamplist()
    if i % 20 == 0:
        patcher.save(version = version, i = i)





values, coefficients, logits = interpolated_path_patch(model)

s = logits.abs().sum()
s.backward()

for i, c in enumerate(coefficients):
    print(i)
    print(c.grad)
for i, v in enumerate(values):
    print(i)
    print(v)
# %%



# model.run_with_hooks(
#     einops.repeat(
#         ioi_cache["resid_pre", 0].unsqueeze(0),
#         "1 ... -> batch2 ...", batch2=2
#     ),
#     start_at_layer=0,
# )

# %%
