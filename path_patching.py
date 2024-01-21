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
# from arena_utils.plotly_utils import imshow, line, scatter, bar
from functools import partial
import einops
import torch.nn.functional as F
# device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

# %%
from parameters import InterpolatedPathPatch, parallelize
# %%

from load_run import model, ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric

# class PathPatchExperiment:
#     def  __init__(self, model, ioi_dataset, abc_dataset):
#         self.model = model
#         self.ioi_dataset = ioi_dataset
#         self.abc_dataset = abc_dataset
#         self.ioi_cache = None
#         self.abc_cache = None
#         self.ioi_metric = None




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











def head_path_patch_hook(
    heads_output :Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    clean_cache: ActivationCache,
    corrupted_activations :Dict[Tuple, Float[Tensor, "batch pos d_head"]],
    ignore_positions : Optional[Bool[Tensor, "layer pos head"]] = None,
) -> Float[Tensor, "batch pos head_index d_head"]:
    # print("called hook", hook.name, hook.layer())
    # replace with clean cache everywhere that's not ignore
    heads_output[:, ~ignore_positions[hook.layer()], :] = (
        clean_cache[hook.name][:, ~ignore_positions[hook.layer()], :]
    )
    for loc, act in corrupted_activations.items():
        layer, head = loc
        # print("searching for", loc)
        if layer == hook.layer():
            # print("corrupted", loc)
            heads_output[:, :, head, :] = act
    # then replace with locations in corrupted activations
    
    return heads_output

def get_path_patch_heads(
    model: HookedTransformer,
    patching_metric: Callable,
    paths : Tuple[Tuple[int, int], Tuple[int, int]],
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
    
) -> Float[Tensor, "layer head"]:

    #step 2
    unique_dest_nodes = set([path_dest for path_src, path_dest in paths])
    unique_dest_nodes = {
        path_dest : [
            path_src for path_src, path_dest2 in paths if path_dest2 == path_dest
        ]
        for path_dest in unique_dest_nodes
    }
    seq_len = new_cache["z", 0].shape[1]
    corrupted_dests = {}
    for path_dest in unique_dest_nodes:
        path_srcs = unique_dest_nodes[path_dest]
        path_dest_layer, path_dest_head = path_dest
        corrupted_srcs = {
            path_src : 
            new_cache["z", path_src[0]][..., path_src[1], :]
            for path_src in path_srcs
        }

        ignore = torch.zeros((model.cfg.n_layers, seq_len,  model.cfg.n_heads), dtype=torch.bool, device=model.cfg.device)
        ignore[path_dest[0], :, path_dest[1]] = True

        patch_function_step_2 = partial(
            head_path_patch_hook,
            clean_cache=orig_cache,
            corrupted_activations = corrupted_srcs,
            ignore_positions = ignore,
            #call on ALL layers prior to dest layer
        )
        min_path_src_layer = min([path_src[0] for path_src in path_srcs])
        model.reset_hooks(including_permanent=True)
        model.add_hook(lambda name: name.endswith("z"), patch_function_step_2)
            # fwd_hooks=[
            #     (utils.get_act_name("z", layer), patch_function_step_2)
            #     for layer in range(min_path_src_layer, path_dest[0])
            # ]
        
        useless_logits, corrupted_path_cache = model.run_with_cache(
            orig_cache["resid_pre", 0],
            start_at_layer=0
        )
        corrupted_dests[path_dest] = (
            corrupted_path_cache["z", path_dest_layer][..., path_dest_head, :]
        )
        assert not (corrupted_path_cache["z", path_dest_layer][..., path_dest_head, :] == orig_cache["z", path_dest_layer][..., path_dest_head, :]).all()
    #step 3
    ignore = torch.zeros((model.cfg.n_layers, seq_len,  model.cfg.n_heads), dtype=torch.bool, device=model.cfg.device)
    # print(corrupted_dests.keys())
    for path_dest in corrupted_dests.keys():
        ignore[path_dest[0], :, path_dest[1]] = True
    patch_function = partial(
        head_path_patch_hook, 
        clean_cache=orig_cache,
        corrupted_activations= corrupted_dests,
        ignore_positions=ignore,
    )
    min_path_dest = min([path_dest[0] for path_dest in unique_dest_nodes])
    model.reset_hooks(including_permanent=True)
    # model.add_hook
    useful_logits = model.run_with_hooks(
        orig_cache["resid_pre", 0],
        start_at_layer=0,
        fwd_hooks=[
            (utils.get_act_name("z", layer), patch_function)
            for layer in range(min_path_dest, model.cfg.n_layers)
        ]
    )
    return useful_logits

# results = torch.zeros(12,12, device = model.cfg.device, dtype=torch.float32)
# for layer in range(10):
#     for head in range(12):
#         torch.cuda.empty_cache()
#         result = ioi_metric(get_path_patch_heads(model, ioi_metric, [((layer, head), (10, 7))])).detach()
#         print(f"{layer}.{head}:", result)
#         results[layer, head] = result

# imshow(
#     results * 100,
#     title="Direct effect on logit difference",
#     labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
#     coloraxis=dict(colorbar_ticksuffix = "%"),
#     width=600,
# )















#%%
# SECOND ITERATION WHICH IS ONE TO MANY


[{},{},{},{},{3:None, 4:None}]

def head_path_patch_hook2(
    heads_output :Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    clean_cache: ActivationCache,
    corrupted_activations :Dict[Tuple, Float[Tensor, "batch pos d_head"]],
    save_list : List[Dict[int, Optional[Tensor]]],
    ignore_positions : Optional[Bool[Tensor, "layer pos head"]] = None,
) -> Float[Tensor, "batch pos head_index d_head"]:
    # print("called hook", hook.name, hook.layer())
    # replace with clean cache everywhere that's not ignore
    for head in save_list[hook.layer()]:
        # print(save_list[hook.layer()])
        if save_list[hook.layer()][head] is None:
            save_list[hook.layer()][head] = heads_output[:, :, head, :].clone()
        else:
            raise ValueError("head already saved")


    heads_output[:, ~ignore_positions[hook.layer()], :] = (
        clean_cache[hook.name][:, ~ignore_positions[hook.layer()], :]
    )
    for loc, act in corrupted_activations.items():
        layer, head = loc
        # print("searching for", loc)
        if layer == hook.layer():
            # print("corrupted2", loc)
            heads_output[:, :, head, :] = act
    # then replace with locations in corrupted activations
    
    return heads_output

def get_path_patch_heads_multidest(
    model: HookedTransformer,
    patching_metric: Callable,
    src_nodes : List[Tuple[int, int]],
    dest_nodes : List[Tuple[int, int]],
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
) -> Float[Tensor, "layer head"]:

    #step 2
    seq_len = new_cache["z", 0].shape[1]
    corrupted_dests = {}
    save_list = [
        {
            head : None
            for head in range(model.cfg.n_heads)
            if (layer, head) in dest_nodes
        }
        for layer in range(model.cfg.n_layers)
    ]
    corrupted_srcs = {
        path_src : 
        new_cache["z", path_src[0]][..., path_src[1], :]
        for path_src in src_nodes
    }

    ignore = torch.zeros((model.cfg.n_layers, seq_len,  model.cfg.n_heads), dtype=torch.bool, device=model.cfg.device)
    patch_function_step_2 = partial(
        head_path_patch_hook2,
        clean_cache=orig_cache,
        corrupted_activations = corrupted_srcs,
        ignore_positions = ignore,
        save_list = save_list,
    )
    # min_path_src_layer = min([path_src[0] for path_src in src_nodes])
    model.reset_hooks()
    model.add_hook(lambda name: name.endswith("z"), patch_function_step_2)
    useless_logits, corrupted_path_cache = model.run_with_cache(
        orig_cache["resid_pre", 0],
        start_at_layer=0
    )
    for i in range(model.cfg.n_layers):
        for head in save_list[i]:
            if save_list[i][head] is not None:
                corrupted_dests[(i, head)] = save_list[i][head]
    #step 3
    ignore = torch.ones((model.cfg.n_layers, seq_len,  model.cfg.n_heads), dtype=torch.bool, device=model.cfg.device)
    patch_function = partial(
        head_path_patch_hook, 
        clean_cache=orig_cache,
        corrupted_activations= corrupted_dests,
        ignore_positions=ignore,
    )
    min_path_dest = min([path_dest[0] for path_dest in dest_nodes])
    model.reset_hooks()
    # model.add_hook
    useful_logits = model.run_with_hooks(
        orig_cache["resid_pre", 0],
        start_at_layer=0,
        fwd_hooks=[
            (utils.get_act_name("z", layer), patch_function)
            for layer in range(min_path_dest, model.cfg.n_layers)
        ]
    )
    return useful_logits


#%%




def head_path_patch_hook2(
    heads_output :Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    clean_cache: ActivationCache,
    corrupted_activations :Dict[Tuple, Float[Tensor, "batch pos d_head"]],
    save_list : List[Dict[int, Optional[Tensor]]],
    ignore_positions : Optional[Bool[Tensor, "layer pos head"]] = None,
) -> Float[Tensor, "batch pos head_index d_head"]:
    # print("called hook", hook.name, hook.layer())
    # replace with clean cache everywhere that's not ignore
    for head in save_list[hook.layer()]:
        # print(save_list[hook.layer()])
        if save_list[hook.layer()][head] is None:
            save_list[hook.layer()][head] = heads_output[:, :, head, :].clone()
        else:
            raise ValueError("head already saved")


    heads_output[:, ~ignore_positions[hook.layer()], :] = (
        clean_cache[hook.name][:, ~ignore_positions[hook.layer()], :]
    )
    for loc, act in corrupted_activations.items():
        layer, head = loc
        # print("searching for", loc)
        if layer == hook.layer():
            # print("corrupted2", loc)
            heads_output[:, :, head, :] = act
    # then replace with locations in corrupted activations
    
    return heads_output

def get_path_patch_heads_multidest(
    model: HookedTransformer,
    patching_metric: Callable,
    src_nodes : List[Tuple[int, int]],
    dest_nodes : List[Tuple[int, int]],
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
) -> Float[Tensor, "layer head"]:

    #step 2
    seq_len = new_cache["z", 0].shape[1]
    corrupted_dests = {}
    save_list = [
        {
            head : None
            for head in range(model.cfg.n_heads)
            if (layer, head) in dest_nodes
        }
        for layer in range(model.cfg.n_layers)
    ]
    corrupted_srcs = {
        path_src : 
        new_cache["z", path_src[0]][..., path_src[1], :]
        for path_src in src_nodes
    }

    ignore = torch.zeros((model.cfg.n_layers, seq_len,  model.cfg.n_heads), dtype=torch.bool, device=model.cfg.device)
    patch_function_step_2 = partial(
        head_path_patch_hook2,
        clean_cache=orig_cache,
        corrupted_activations = corrupted_srcs,
        ignore_positions = ignore,
        save_list = save_list,
    )
    # min_path_src_layer = min([path_src[0] for path_src in src_nodes])
    model.reset_hooks()
    model.add_hook(lambda name: name.endswith("z"), patch_function_step_2)
    useless_logits, corrupted_path_cache = model.run_with_cache(
        orig_cache["resid_pre", 0],
        start_at_layer=0
    )
    for i in range(model.cfg.n_layers):
        for head in save_list[i]:
            if save_list[i][head] is not None:
                corrupted_dests[(i, head)] = save_list[i][head]
    #step 3
    ignore = torch.ones((model.cfg.n_layers, seq_len,  model.cfg.n_heads), dtype=torch.bool, device=model.cfg.device)
    patch_function = partial(
        head_path_patch_hook, 
        clean_cache=orig_cache,
        corrupted_activations= corrupted_dests,
        ignore_positions=ignore,
    )
    min_path_dest = min([path_dest[0] for path_dest in dest_nodes])
    model.reset_hooks()
    # model.add_hook
    useful_logits = model.run_with_hooks(
        orig_cache["resid_pre", 0],
        start_at_layer=0,
        fwd_hooks=[
            (utils.get_act_name("z", layer), patch_function)
            for layer in range(min_path_dest, model.cfg.n_layers)
        ]
    )
    return useful_logits




#%%

# I expect this to take way too much memory
# but then we can pair down the graph and then do this method


def get_patch_coefficients(model: HookedTransformer):
    patch_coefficients = [ 
        torch.ones(
            model.cfg.n_heads,
            device=model.cfg.device,
            requires_grad=True
        )
    ] + [
        torch.rand(
            (
                model.cfg.n_heads, 
                layer, 
                model.cfg.n_heads
            ),
            device=model.cfg.device,
            requires_grad=True
        )
        for layer in range(1, model.cfg.n_layers)
    ] + [
        torch.rand(
            (
                model.cfg.n_heads, 
                model.cfg.n_heads
            ),
            device=model.cfg.device,
            requires_grad=True
        )
    ]
    return patch_coefficients


def get_per_layer_patch_coefficients(model: HookedTransformer):
    roots = [ 
        torch.ones(
            model.cfg.n_heads,
            device=model.cfg.device,
            requires_grad=True
        )
    ] + [
        torch.rand(
            (
                1, 
                layer, 
                model.cfg.n_heads
            ),
            device=model.cfg.device,
            requires_grad=True
        )
        for layer in range(1, model.cfg.n_layers)
    ] + [
        torch.rand(
            (
                model.cfg.n_heads, 
                model.cfg.n_heads
            ),
            device=model.cfg.device,
            requires_grad=True
        )
    ]
    patch_coefficients = [roots[0]] + [
        parallelize(
            root,
            model.cfg.n_heads
        )
        for root in roots[1:-1]
    ] + [roots[-1]]

    return patch_coefficients, roots


def clean_dirty_resid_pre(
    resid_head_dirty_coeffs : Union[
        Float[Tensor, "head_index"],
        Float[Tensor, "pos head_index"] # in this this mode you can see which heads need to read from where!
    ],
    clean_cache: ActivationCache,
    dirty_cache: ActivationCache,
    parallelism = 1
):
    if parallelism == 1:
        return clean_cache["resid_pre", 0].lerp(
            dirty_cache["resid_pre", 0], 
            resid_head_dirty_coeffs.unsqueeze(-1)
        )
    else:
        batch_size = clean_cache["resid_pre", 0].shape[0]
        segment_size = batch_size // parallelism
        assert batch_size % parallelism == 0
        outs = []
        for i in range(parallelism):
            out = clean_cache["resid_pre", 0].lerp(
                dirty_cache["resid_pre", 0][i * segment_size : (i + 1) * segment_size], 
                resid_head_dirty_coeffs[i].unsqueeze(-1)
            )
            outs.append(out)
        out = torch.cat(outs, dim=0)
        return out

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
        out = clean_cache[hook.name].lerp(
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
            out = clean_cache[hook.name].lerp(
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
    patch_coefficients : Optional[List[Float[Tensor, "... head_index"]]] = None,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
    parallelism = 1
) -> Float[Tensor, "layer head"]:

    #step 2
    seq_len = new_cache["z", 0].shape[1]
    batch_size = new_cache["z", 0].shape[0]

    if patch_coefficients is None:
        patch_coefficients = get_patch_coefficients()
    patched_heads_values = [
        orig_cache["z", 0].lerp(
            new_cache["z", 0], patch_coefficients[0].unsqueeze(-1)
        )
    ]
    orig_cache["resid_pre", 0].shape
    orig_resid_pre = parallelize(orig_cache["resid_pre", 0], parallelism)
    # einops.repeat(
    #     orig_cache["resid_pre", 0],
    #     "batch ... -> (parallelism batch) ...",
    #     parallelism=parallelism
    # )
    # if parallelism != 1:
    #     _, orig_cache = model.run_with_cache(
    #         orig_resid_pre,
    #         start_at_layer=0
    #     )
    # print("orig_resid_pre", orig_cache["resid_pre", 0][0] == orig_resid_pre[1])
    # print(orig_cache["resid_pre", 0][0] == orig_resid_pre[0 + parallelism])
    # print(orig_cache["resid_pre", 0][0] == orig_resid_pre[0 + batch_size])

    # print(orig_cache["resid_pre", 0].shape)
    # print(orig_resid_pre.shape)
    for layer in range(1, model.cfg.n_layers):
        patched_heads_in_layer = []
        print("\nlayer", layer, end="->")
        for head_i in range(model.cfg.n_heads // parallelism):  
            head = head_i * parallelism
            print(head, end=", ")
            print("\n patch_shape[layer]", patch_coefficients[layer].shape)
            if parallelism == 1:
                head_coeffs = patch_coefficients[layer][head]
            else:
                head_coeffs = patch_coefficients[layer][head : head + parallelism]
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
                orig_resid_pre,
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
        zcat = torch.cat(patched_heads_in_layer, dim=-2)
        patched_heads_values += [zcat]
    last_patch_coeffs = patch_coefficients[-1]
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

    # model.reset_hooks()
    # model.add_hook(
    #     lambda name: name.endswith("z") and not f".{layer}." in name,
    #     hook_fn
    # )
    # model.add_hook(
    #     lambda name: name.endswith("z") and f".{layer}." in name,
    #     extract_hook_fn
    # )
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


patcher = InterpolatedPathPatch(model)
# patch_coefficients = [patcher.pre] + [*patcher.layers[1:]] + [patcher.post]
optim = torch.optim.Adam(patcher.parameters(), lr=0.1, weight_decay=0, betas=(0.9, 0.99))
# patch_coefficients, roots = get_per_layer_patch_coefficients(model)
# optim = torch.optim.Adam(roots, lr=0.1, weight_decay=0.01)
# optim = torch.optim.Adam(roots, lr=0.1, weight_decay=0, betas=(0.9, 0.94))
# clamped_coefficients = patcher.clamplist()
patch_coefficients = [patcher.pre] + [*patcher.layers[1:]] + [patcher.post]
clamped_coefficients = gradthruclamplist(patch_coefficients)


for i in range(100):
    print("step ", i)
    optim.zero_grad()
    values, coefficients, logits = interpolated_path_patch(
        model, 
        coeffs=patcher,
        patch_coefficients=clamped_coefficients,
        parallelism=12
    )
    # patch_coefficients = patcher.tolist()
    loss_l1 = 1 / 3 * (
        0.011   * sum([(c.sum() + torch.relu(c).sum()) * 0.5 for c in patch_coefficients])
        - 0.001 * sum([(c.sum() + torch.relu(c).sum()) * 0.5 for c in patch_coefficients[1:-1]])
    )
    loss_logits = ioi_metric(logits)
    loss = loss_logits + loss_l1
    l0 = sum([torch.count_nonzero(c) for c in clamped_coefficients])
    if l0 < 100:
        patcher.print_connections()
    print("losses:", loss_logits.item(), loss_l1.item())
    print("nonzero coeffs:", l0)
    loss.backward()
    optim.step()
    # clamped_coefficients = patcher.clamplist()
    clamped_coefficients = gradthruclamplist(patch_coefficients)



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
