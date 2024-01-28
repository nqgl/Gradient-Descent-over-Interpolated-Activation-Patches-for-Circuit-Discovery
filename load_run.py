from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor

from ioi_dataset import NAMES, IOIDataset
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float, Int, Bool
import torch
t = torch
import re
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
import torch.nn.functional as F
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
# model = torch.compile(model)

# def format_prompt(sentence: str) -> str:
#     '''Format a prompt by underlining names (for rich print)'''
#     return re.sub("(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence) + "\n"


# def make_table(cols, colnames, title="", n_rows=5, decimals=4):
#     '''Makes and displays a table, from cols rather than rows (using rich print)'''
#     table = Table(*colnames, title=title)
#     rows = list(zip(*cols))
#     f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
#     for row in rows[:n_rows]:
#         table.add_row(*list(map(f, row)))
# #     rprint(table)


# make_table(
#     colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
#     cols = [
#         map(format_prompt, ioi_dataset.sentences), 
#         model.to_string(ioi_dataset.s_tokenIDs).split(), 
#         model.to_string(ioi_dataset.io_tokenIDs).split(), 
#         map(format_prompt, abc_dataset.sentences), 
#     ],
#     title = "Sentences from IOI vs ABC distribution",
# )

import time
def get_next_data(N=1, model=model, seed=None):
    if seed is None:
        seed = int(time.time())
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device=str(device)
    )
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")



    ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
    abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)


    def bind_metrics_to_dataset(model :HookedTransformer, ioi_dataset :IOIDataset, abc_dataset :IOIDataset) -> Dict[str, List[float]]:
        model.reset_hooks(including_permanent=True)




        def kl_divergence_metric(
            logits: Float[Tensor, "batch seq d_vocab"],
            ioi_dataset: IOIDataset = ioi_dataset,
            abc_dataset: IOIDataset = abc_dataset,
        ) -> float:
            '''
            Returns the KL divergence of logits with the ABC distribution.
            '''
            # Only the final logits are relevant for the answer
            assert logits.size(0) == ioi_dataset.N
            end_logits = logits[range(logits.size(0)), ioi_dataset.word_idx["end"]] # [batch, vocab]
            abc_logits = abc_logits_original[range(abc_logits_original.size(0)), ioi_dataset.word_idx["end"]] # [batch, vocab]

            # Find KL divergence
            return torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(end_logits, dim=-1),
                torch.nn.functional.log_softmax(abc_logits, dim=-1),
                reduction="batchmean",
                log_target=True
            )

        def logits_to_ave_logit_diff_2(
            logits: Float[Tensor, "batch seq d_vocab"],
            ioi_dataset: IOIDataset = ioi_dataset,
            per_prompt=False
        ) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
            '''
            Returns logit difference between the correct and incorrect answer.

            If per_prompt=True, return the array of differences rather than the average.
            '''
            logits = F.log_softmax(logits, dim=-1)
            # Only the final logits are relevant for the answer
            # Get the logits corresponding to the indirect object / subject tokens respectively
            io_logits = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs] # [batch]
            s_logits = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs] # [batch]
            # Find logit difference
            answer_logit_diff = io_logits - s_logits
            return answer_logit_diff if per_prompt else answer_logit_diff.mean()

        # ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
        # abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

        ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
        abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()




        def ioi_metric_2(
            logits: Float[Tensor, "batch seq d_vocab"],
            clean_logit_diff: float = ioi_average_logit_diff,
            corrupted_logit_diff: float = abc_average_logit_diff,
            ioi_dataset: IOIDataset = ioi_dataset,
        ) -> float:
            '''
            We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset), 
            and -1 when performance has been destroyed (i.e. is same as ABC dataset).
            '''
            patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
            return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

        return ioi_metric_2, kl_divergence_metric

    ioi_metric, kl_metric = bind_metrics_to_dataset(model, ioi_dataset, abc_dataset)
    return ioi_cache, abc_cache, ioi_metric, kl_metric

