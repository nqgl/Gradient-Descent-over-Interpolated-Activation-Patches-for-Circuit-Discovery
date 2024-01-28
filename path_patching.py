import torch

from parameters import InterpolatedPathPatch

from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from ioi_dataset import IOIDataset
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from functools import partial

from parameters import InterpolatedPathPatch, mylerp

from nqgl.mlutils.norepr import fastpartial as partial


def extract_tensor_hook(
    heads_output: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    heads: List,
    head: Optional[int],
    parallelism=1,
):
    if parallelism == 1:
        if head is None:
            heads.append(heads_output)
        else:
            heads.append(heads_output[:, :, head, :].unsqueeze(-2))
    else:
        batch_size = heads_output.shape[0]
        segment_size = batch_size // parallelism
        outs = []
        assert head is not None
        for i in range(parallelism):
            heads.append(
                heads_output[
                    i * segment_size : (i + 1) * segment_size, :, head + i, :
                ].unsqueeze(-2)
            )
    return heads_output


def head_interpolate_hook(
    heads_output: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    clean_cache: ActivationCache,
    patched_heads: Float[Tensor, "layer batch pos head_index d_head"],
    patch_coefficients: Float[Tensor, "run_layers head_index"],
    coeffs: InterpolatedPathPatch,
    parallelism=1,
) -> Float[Tensor, "batch pos head_index d_head"]:
    if parallelism == 1:
        return mylerp(
            clean_cache[hook.name],
            patched_heads[hook.layer()],
            patch_coefficients[hook.layer()].unsqueeze(-1),
        )
    batch_size = heads_output.shape[0]

    segment_size = batch_size // parallelism
    assert batch_size % parallelism == 0
    outs = []
    for i in range(parallelism):
        out = mylerp(
            clean_cache[hook.name],
            patched_heads[hook.layer()],
            patch_coefficients[i][hook.layer()].unsqueeze(-1),
        )
        outs.append(out)
    out = torch.cat(outs, dim=0)
    return out


def interpolated_path_patch(
    model: HookedTransformer,
    coeffs: InterpolatedPathPatch,
    new_cache: Optional[ActivationCache],
    orig_cache: Optional[ActivationCache],
    parallelism=1,
) -> Float[Tensor, "layer head"]:
    patch_coefficients = coeffs.c_layers()
    patched_heads_values = []
    for layer in range(0, model.cfg.n_layers):
        patched_heads_in_layer = []
        for head_i in range(model.cfg.n_heads // parallelism):
            head = head_i * parallelism
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
                coeffs=coeffs,
                patched_heads=patched_heads_values,
                # ignore_positions = None,
                patch_coefficients=head_coeffs,
                parallelism=parallelism,
            )

            extract_hook_fn = partial(
                extract_tensor_hook,
                heads=patched_heads_in_layer,
                head=head,
                parallelism=parallelism,
            )
            logits = model.run_with_hooks(
                coeffs.interpolate_resid_pres(
                    orig_cache["resid_pre", 0],
                    new_cache["resid_pre", 0],
                    layer=layer,
                    parallelism=parallelism,
                    start_head=head if parallelism != model.cfg.n_heads else None,
                ),
                start_at_layer=0,
                stop_at_layer=layer + 1,
                fwd_hooks=[
                    (
                        lambda name: name.endswith("z") and not f".{layer}." in name,
                        hook_fn,
                    ),
                    (
                        lambda name: name.endswith("z") and f".{layer}." in name,
                        extract_hook_fn,
                    ),
                ],
            )
        zcat = torch.cat(patched_heads_in_layer, dim=-2)
        patched_heads_values += [zcat]
    last_patch_coeffs = coeffs.c_out()
    hook_fn = partial(
        head_interpolate_hook,
        clean_cache=orig_cache,
        coeffs=coeffs,
        patched_heads=patched_heads_values,
        patch_coefficients=last_patch_coeffs,
        parallelism=1,
    )

    extract_hook_fn = partial(
        extract_tensor_hook, heads=patched_heads_in_layer, head=head, parallelism=1
    )

    logits = model.run_with_hooks(
        orig_cache["resid_pre", 0],
        start_at_layer=0,
        fwd_hooks=[(lambda name: name.endswith("z"), hook_fn)],
    )

    return patched_heads_values, patch_coefficients, logits
