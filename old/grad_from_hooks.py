import transformer_lens
import circuitsvis as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from jaxtyping import Float
import transformer_lens.utils as utils
import ioi_prompts  
# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

#TODO clean up all of the unwrapped closures by wrapping them, and then move all of the top level state code into main()

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")
print(activations["blocks.0.attn.hook_v"].shape)
class GradThruRelu(Function):
    @staticmethod
    def forward(ctx, x):
        return F.relu(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
gradthrurelu = GradThruRelu.apply
def gradthruclamp(x):
    return 1 - gradthrurelu(1 - gradthrurelu(x))


n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
#C_ablation = torch.zeros((n_layers, n_heads))
# C_ablations = {}
prompts, corrupted_prompts, answers, answer_tokens, corrupted_tokens = ioi_prompts.prompts(model)
tokens = model.to_tokens(prompts, prepend_bos=True)
seq_len = tokens.shape[1]
torch.manual_seed(3)
ablation_parameters = torch.rand((n_layers, 1, seq_len, n_heads, 1)) / 10
ablation_parameters = torch.rand((n_layers, 1, 1, n_heads, 1)) / 1
print(ablation_parameters.shape)
ablation_parameters.requires_grad = True


def simple_value_mutator(value, layer_ablation_parameters, layer_index):
    return value * (1 - layer_ablation_parameters.view(1, 1, -1, 1))


def create_ablation_model(model, ablation_parameters, value_mutator):
    hook_tuples = head_ablation_hook_tuples(ablation_parameters=ablation_parameters, value_mutator=value_mutator)
    def _hooked_run(tokens, return_type="logits", **kwargs):
        return model.run_with_hooks(tokens, return_type=return_type, fwd_hooks=hook_tuples, **kwargs)
    return _hooked_run

def head_ablation_hook_tuples(ablation_parameters, value_mutator=simple_value_mutator):
    def head_ablation_hook_generator(layer_ablation_parameters: Float[torch.Tensor, "head_index"], layer_index: int):
        def head_ablation_hook_fn(
                    value: Float[torch.Tensor, "batch pos head_index d_head"],
                    hook: transformer_lens.hook_points.HookPoint
                ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            # print(f"Shape of the value tensor: {value.shape}")
            return value_mutator(value, gradthruclamp(layer_ablation_parameters.to(value.device)), layer_index)
        return head_ablation_hook_fn
    
    hook_tuples = [(
        utils.get_act_name("v", layer_to_ablate), 
        head_ablation_hook_generator(ablation_parameters[layer_to_ablate], layer_to_ablate)
        ) for layer_to_ablate in range(n_layers)]

    return hook_tuples



def patching_value_mutator_generator(corrupted_activations):
    def patching_value_mutator(value, layer_ablation_parameters, layer_index): #todo, change this to activation_location
        patch_in_activations = corrupted_activations[utils.get_act_name("v", layer_index)]
        # print(patch_in_activations.shape)
        # print(value.shape)
        # layer_ablation_parameters = layer_ablation_parameters.view(1, layer_ablation_parameters.shape[1], -1, 1)
        # print(value.shape, layer_ablation_parameters.shape, patch_in_activations.shape)
        return value * (1 - layer_ablation_parameters) + layer_ablation_parameters * patch_in_activations
    return patching_value_mutator


def logit_cross_entropy_loss(logits_1, logits_2):
    return F.cross_entropy(logits_1, F.softmax(logits_2, dim=-1))

def logit_kl_loss(logits_1, logits_2):
    return F.kl_div(F.log_softmax(logits_1, dim=-1), F.log_softmax(logits_2, dim=-1), log_target=True, reduction="batchmean")

def penalty_function_l_half(ablation_parameters):
    n = torch.count_nonzero(torch.relu(ablation_parameters - 0.04))
    return torch.mean(torch.sqrt(torch.abs(ablation_parameters) + 0.0001)) * n

def penalty_function_l1(ablation_parameters, nonzero_penalty_n = True):
    if nonzero_penalty_n is False:
        n = 1
    else:
        if nonzero_penalty_n is True:
            nonzero_penalty_n = 0
        n = torch.count_nonzero(torch.relu(ablation_parameters - 0.04)) - nonzero_penalty_n
        n = max(n, 1)
    return torch.mean(torch.abs(ablation_parameters)) * n

def loss_function_minimal_sufficient(target_logits, ablated_logits, ablation_parameters, penalty_function = penalty_function_l1):
    ablation_parameters = gradthruclamp(ablation_parameters)
    # print(ablation_parameters)
    ce_loss = logit_kl_loss(target_logits, ablated_logits)
    penalty = penalty_function(ablation_parameters)
    if ablation_parameters.shape[2] > 1:
        penalty2 = 0
        for i in range(ablation_parameters.shape[2]):
            penalty2 += penalty_function(ablation_parameters[:, :, i:i+1, :, :])
        penalty = penalty2 * 10
    # print(f"ce_loss: {ce_loss}, penalty: {penalty}")
    return ce_loss + penalty * 0.01
    
def loss_function_maximal_sufficient(target_logits, ablated_logist, ablation_parameters):
    return loss_function_minimal_sufficient(target_logits, ablated_logist, 1 - ablation_parameters)

def ablation_masker(ablation_parameters, head_tuples):
    for head_tuple in head_tuples:
        ablation_parameters[head_tuple[0], 0, 0, head_tuple[1], 0] = 0
    return ablation_parameters
torch.manual_seed(2)
# ablation_parameters = torch.rand(n_layers, n_heads) / 1
# ablation_parameters.requires_grad = True

loss_function = loss_function_maximal_sufficient
loss_function = loss_function_minimal_sufficient

prompts, corrupted_prompts, answers, answer_tokens, corrupted_tokens = ioi_prompts.prompts(model)

corrupted_logits, corrupted_activations = model.run_with_cache(corrupted_tokens, return_type="logits")

patching_value_mutator = patching_value_mutator_generator(corrupted_activations)
patched_model = create_ablation_model(model, ablation_parameters, patching_value_mutator)


tokens = model.to_tokens(prompts, prepend_bos=True)

ablated_loss = patched_model(tokens, return_type="loss")


optimizer = torch.optim.SGD([ablation_parameters], lr=0.2, momentum=0.95)
torch.set_printoptions(sci_mode=False, linewidth=120, precision=3)
consecutive_zero_intermediates = 0
for i in range(400):
    # original_logits = model(tokens, return_type="logits")
    corrupted_logits = model(corrupted_tokens, return_type="logits")
    original_logits = model(tokens, return_type="logits")
    ablated_logits = patched_model(tokens, return_type="logits")
    ablated_loss = patched_model(tokens, return_type="loss")
    optimizer.zero_grad()
    # print(corrupted_logits.shape)
    ablation_parameters_dropped = F.dropout(ablation_parameters, p=0.5, training=True)
    loss = loss_function(original_logits[:, -1, :], ablated_logits[:, -1, :], ablation_parameters_dropped)
    
    # loss = loss_function(original_logits[:, -1, answer_tokens], ablated_logits[:, -1, answer_tokens], ablation_parameters_dropped)
    # loss = loss_function(original_logits[:, -1, answer_tokens], ablated_logits[:, -1, answer_tokens], ablation_parameters_dropped)
    
    loss = loss_function(corrupted_logits[:, -1, answer_tokens], ablated_logits[:, -1, answer_tokens], ablation_parameters_dropped)
    # loss = 0
    # loss += loss_function(corrupted_logits[:, -1, :], ablated_logits[:, -1, :], ablation_parameters_dropped)
    
    # extremified = torch.zeros_like(corrupted_logits[:, -1, answer_tokens])
    # for i in range(extremified.shape[0]):
    #     extremified[i, (i) % 2] = 10000
    # loss = loss_function(ablated_logits[:, -1, answer_tokens], extremified,  ablation_parameters_dropped)

    loss.backward()
    optimizer.step()
    # print(ablation_parameters)
    print(f"Ablated loss:{ablated_loss}")
    intermediates = torch.logical_and(ablation_parameters > 0, ablation_parameters < 1)
    c_intermediates = torch.count_nonzero(intermediates)
    print(f"Intermediate ablation parameters: {c_intermediates}")
    print(f"nnm: {ablation_parameters[10, 0, 0, 7, 0]}, {ablation_parameters[11, 0, 0, 10, 0]}")
    if c_intermediates != 0:
        consecutive_zero_intermediates = 0
    else:
        consecutive_zero_intermediates += 1
        if consecutive_zero_intermediates > 5:
            break

def show_prompt_differences(patched_model, model, prompt_ids=None):
    prompts, corrupted_prompts, answers, answer_tokens, corrupted_tokens = ioi_prompts.prompts(model)
    corrupted_logits, corrupted_activations = model.run_with_cache(corrupted_tokens, return_type="logits")
    tokens = model.to_tokens(prompts, prepend_bos=True)
    ablated_logits = patched_model(tokens)
    prompt_ids = range(len(prompts)) if prompt_ids is None else prompt_ids
    for i in prompt_ids:
        prompt_ablated_logits = ablated_logits[i:i + 1, :, :]
        prompt = prompts[i]
        tokens = model.to_tokens(prompt, prepend_bos=True)
        original_logits = model(tokens, return_type="logits")
        print(f"\nPrompt: {prompt}")
        print(f"Answers: {answers[i]}")
        p_original = F.softmax(original_logits[:, -1, :], dim=-1)
        p_answers_original = [p_original[0, answer_tokens[i, 0]], p_original[0, answer_tokens[i, 1]]] 
        p_ablated = F.softmax(prompt_ablated_logits[:, -1, :], dim=-1)
        p_answers_ablated = [p_ablated[0, answer_tokens[i, 0]], p_ablated[0, answer_tokens[i, 1]]]
        print(f"Answer \"{answers[i][0]}\" original_prob: {p_answers_original[0].item()}, ablated_prob: {p_answers_ablated[0].item()}")
        print(f"Answer \"{answers[i][1]}\" original_prob: {p_answers_original[1].item()}, ablated_prob: {p_answers_ablated[1].item()}")
        top_token_original = torch.argmax(original_logits[:, -1, :], dim=-1)
        top_token_ablated = torch.argmax(prompt_ablated_logits[:, -1, :], dim=-1)
        print(f"Top token original: {model.tokenizer.decode(top_token_original.item())}, ablated: {model.tokenizer.decode(top_token_ablated.item())}")
        k = 5
        topk_original = torch.topk(original_logits[:, -1, :], k, dim=-1)[1][0]
        topk_ablated = torch.topk(prompt_ablated_logits[:, -1, :], k, dim=-1)[1][0]
        k_sp_original = [(model.tokenizer.decode(t.item()), p_original[0, t], p_ablated[0, t]) for t in topk_original]
        k_sp_ablated = [(model.tokenizer.decode(t.item()), p_original[0, t], p_ablated[0, t]) for t in topk_ablated]
        top_original_str = "\n\t".join([f'"{t[0]}" original_prob: {t[1].item()}, \t ablated_prob: {t[2].item()}' for i, t in enumerate(k_sp_original)])
        top_ablated_str = "\n\t".join([f'"{t[0]}" , ablated_prob: {t[2].item()} \t original_prob: {t[1].item()}' for i, t in enumerate(k_sp_ablated)])
        print(f"Top {k} tokens original: \n\t{top_original_str}")
        print(f"Top {k} tokens ablated: \n\t{top_ablated_str}")





def show_top_k_next_tokens(patched_model, model, prompts, graph=False):
    import matplotlib.pyplot as plt
    tokens = model.to_tokens(prompts, prepend_bos=True)
    ablated_logits = patched_model(tokens)
    for i in range(len(prompts)):
        prompt_ablated_logits = ablated_logits[i:i + 1, :, :]
        prompt = prompts[i]
        tokens = model.to_tokens(prompt, prepend_bos=True)
        original_logits = model(tokens, return_type="logits")
        # print(f"\nPrompt: {prompt}")
        p_original = F.softmax(original_logits[:, -1, :], dim=-1)
        p_ablated = F.softmax(prompt_ablated_logits[:, -1, :], dim=-1)
        top_token_original = torch.argmax(original_logits[:, -1, :], dim=-1)
        top_token_ablated = torch.argmax(prompt_ablated_logits[:, -1, :], dim=-1)
        # print(f"Top token original: {model.tokenizer.decode(top_token_original.item())}, ablated: {model.tokenizer.decode(top_token_ablated.item())}")
        k = 5
        topk_original = torch.topk(original_logits[:, -1, :], k, dim=-1)[1][0]
        topk_ablated = torch.topk(prompt_ablated_logits[:, -1, :], k, dim=-1)[1][0]
        k_sp_original = [(model.tokenizer.decode(t.item()), p_original[0, t], p_ablated[0, t]) for t in topk_original]
        k_sp_ablated = [(model.tokenizer.decode(t.item()), p_original[0, t], p_ablated[0, t]) for t in topk_ablated]
        top_original_str = "\n\t".join([f'"{t[0]}" original_prob: {t[1].item()}, \t ablated_prob: {t[2].item()}' for i, t in enumerate(k_sp_original)])
        top_ablated_str = "\n\t".join([f'"{t[0]}" , ablated_prob: {t[2].item()} \t original_prob: {t[1].item()}' for i, t in enumerate(k_sp_ablated)])
        # print(f"Top {k} tokens original: \n\t{top_original_str}")
        # print(f"Top {k} tokens ablated: \n\t{top_ablated_str}")

        if graph:
            import numpy as np
            barWidth = 0.35
            for entry in k_sp_original:
                if entry[0] in [t[0] for t in k_sp_ablated]:
                    k_sp_ablated.pop([t[0] for t in k_sp_ablated].index(entry[0]))
            k_sp_ablated.reverse()
            n = len(k_sp_ablated + k_sp_original)
            r1 = np.arange(n)
            r2 = [x + barWidth for x in r1]

            plt.figure()
            plt.title(f"Top next word probabilities distribution for prompt =\n\"{prompt}\"")
            plt.bar(r1, [t[1].item() for t in k_sp_original] + [t[1].item() for t in k_sp_ablated], width=barWidth, label="original")
            plt.bar(r2, [t[2].item() for t in k_sp_original] + [t[2].item() for t in k_sp_ablated], width=barWidth, label="patched")
            plt.xlabel('Tokens')
            plt.xticks([r + barWidth / 2 for r in range(n)], [t[0] for t in k_sp_original] + [t[0] for t in k_sp_ablated])
            plt.ylabel('Probability')
            plt.legend()
            plt.show()





def make_ablations_binary(ablation_parameters, threshold = 0.5):
    return torch.where(ablation_parameters > 0.5, torch.ones_like(ablation_parameters), torch.zeros_like(ablation_parameters))

# show_prompt_differences(patched_model, model)
print(make_ablations_binary(ablation_parameters))
binary_patched_model = create_ablation_model(model, make_ablations_binary(ablation_parameters), patching_value_mutator)
show_prompt_differences(binary_patched_model, model)
torch.save(ablation_parameters, "ablation_parameters.pt")
print(sum(make_ablations_binary(ablation_parameters).flatten()))

# null_patched_model = create_ablation_model(model, torch.zeros_like(ablation_parameters), patching_value_mutator)
# show_prompt_differences(null_patched_model, model)



def print_heads_copied(ablation_parameters, show_prompt_differences=False):
    patched_vs_unpatched = (ablation_parameters.shape.numel() / 2 > torch.count_nonzero(ablation_parameters > 0.5))
    identified = []
    if ablation_parameters.shape[2] == 1:    
        for layer in range(n_layers):
            for head in range(n_heads):
                if (ablation_parameters[layer, 0, 0, head, 0] > 0.5) ^ (not patched_vs_unpatched):
                    identified.append((layer, head))
                    if patched_vs_unpatched:
                        print(f"Layer {layer} head {head} patched")
                    else:
                        print(f"Layer {layer} head {head} unpatched")
                    

    else:
        for pos in range(ablation_parameters.shape[2]):
            for layer in range(n_layers):
                for head in range(n_heads):
                    if (ablation_parameters[layer, 0, pos, head, 0] > 0.5) ^ (not patched_vs_unpatched):
                        identified.append((layer, head, pos))
                        if patched_vs_unpatched:
                            print(f"Layer {layer} head {head} patched at token {pos}")
                        else:
                            print(f"Layer {layer} head {head} unpatched at token {pos}")
                        x = ablation_parameters.clone()
                        x[layer, 0, pos, head, 0] = 0
                        y = torch.zeros_like(ablation_parameters)
                        y[layer, 0, pos, head, 0] = 1
                        if show_prompt_differences:
                            show_prompt_differences(create_ablation_model(model, x, patching_value_mutator), model, prompt_ids=[0])
                            show_prompt_differences(create_ablation_model(model, y, patching_value_mutator), model, prompt_ids=[0])
    return identified
head_tuples = print_heads_copied(ablation_parameters)
show_top_k_next_tokens(binary_patched_model, model, prompts)
def test_ablation_parameters(ablation_parameters, model, patching_value_mutator, custom=False):
    test_string = "After Kat and Carl went to the bar, Carl gave a ball to"
    test_string2 = "After Kat and Carl went to the bar, Kat gave a ball to"
    prompts = [test_string, test_string2] * 4
    patched_model = create_ablation_model(model, make_ablations_binary(ablation_parameters), patching_value_mutator)
    show_top_k_next_tokens(patched_model, model, prompts)
    show_top_k_next_tokens(patched_model, model, prompts[1:-1])

# test_ablation_parameters(ablation_parameters, model, patching_value_mutator)
# while True:
    # test_ablation_parameters(ablation_parameters, model, patching_value_mutator, custom=True)

def generate_patched_graphs(model, binary_patched_model, nb_patched_model):
    prompts, corrupted_prompts, answers, answer_tokens, corrupted_tokens = ioi_prompts.prompts(model)
    corrupted_logits = model(corrupted_tokens, return_type="logits")
    tokens = model.to_tokens(prompts, prepend_bos=True)
    ablated_logits = binary_patched_model(tokens)
    nb_ablated_logits = nb_patched_model(tokens)
    original_logits = model(tokens, return_type="logits")
    import matplotlib.pyplot as plt

    # bar graph of the probabilities of each answer
    # categories: original, corrupted_prompt, binary ablated, ablated
    


def categorize_heads(head_list, doprint = False):
    # Categories based on the provided data
    categories = {
        "previous token": [(2, 2), (4, 11)],
        "duplicate token": [(0, 1), (3, 0)],
        "induction heads": [(5, 5), (6, 9)],
        "s-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
        "negative name mover": [(10, 7), (11, 10)],
        "name mover": [(9, 9), (9, 6), (10, 0)],
        "backup name mover": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
        "parenthetical heads": [(0, 10), (5, 8), (5, 9)]
    }

    # Initialize counts
    category_counts = {key: 0 for key in categories}
    total_identified = 0
    not_in_circuit = []

    # Process each tuple
    for head in head_list:
        found = False
        for category, heads in categories.items():
            if head in heads:
                category_counts[category] += 1
                total_identified += 1
                found = True
                break
        if not found:
            not_in_circuit.append(head)

    if print:
        for category in categories:
            print(f"{category}: {category_counts[category]} / {len(categories[category])}")
        print(f"Total in circuit: {total_identified}\nTotal: {len(head_list)} ")
        print(f"Not in circuit: {not_in_circuit}")

    return category_counts, total_identified, not_in_circuit

print("Identified heads:")
ident, n_found, n_wrong = categorize_heads(head_tuples, doprint=True)


print("intermediate:")
print_heads_copied(intermediates)