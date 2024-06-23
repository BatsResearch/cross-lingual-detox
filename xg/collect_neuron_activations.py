from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import utils

torch.set_grad_enabled(False)


def collect_neuron_activation_gpt2(
    orig_model,
    dpo_model,
    promopt_tokens: Tensor,
    neuron_list: List[Tuple[int, int]],
    batch_size: int = 10,
    n_new_tokens: int = 20,
):
    """Extract the post activation for selected neurons in `neuron_list` averaged across next
    `n_new_tokens` tokens

    Args:
        orig_model (_type_): model before DPO training
        dpo_model (_type_): model after DPO training
        promopt_tokens (Tensor): _description_
        neuron_list (List[Tuple[int, int]]): list of neurons in format (layer, id) from which activations are collected.
        batch_size (int, optional): batch size during forward pass Defaults to 10.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    promopt_tokens = promopt_tokens.to(device)

    layers_of_interest = [n[0] for n in neuron_list]
    layers_filter = lambda name: name in [
        utils.get_act_name("mlp_post", l) for l in layers_of_interest
    ]

    orig_post_acts = defaultdict(list)
    dpo_post_acts = defaultdict(list)

    for idx in tqdm(range(0, promopt_tokens.shape[0], batch_size)):

        batch = promopt_tokens[idx : idx + batch_size]
        dpo_batch = batch.clone()

        for _ in range(n_new_tokens):
            with torch.inference_mode():
                logits, cache = orig_model.run_with_cache(
                    batch, names_filter=layers_filter
                )

            # next tokens for the whole batch, shape (batch, 1)
            next_tokens: torch.Tensor = logits.argmax(dim=-1)[:, -1]

            for layer, idx in neuron_list:
                post_act = cache[utils.get_act_name("mlp_post", layer)][:, -1, idx]
                orig_post_acts[(layer, idx)].extend(post_act.tolist())

            # append the generated tokens to the batch
            batch = torch.concat([batch, next_tokens.unsqueeze(dim=-1)], dim=-1)

            with torch.inference_mode():
                logits, cache = dpo_model.run_with_cache(
                    dpo_batch, names_filter=layers_filter
                )

            next_tokens: torch.Tensor = logits.argmax(dim=-1)[:, -1]

            for layer, idx in neuron_list:
                post_act = cache[utils.get_act_name("mlp_post", layer)][:, -1, idx]
                dpo_post_acts[(layer, idx)].extend(post_act.tolist())
            dpo_batch = torch.concat([dpo_batch, next_tokens.unsqueeze(dim=-1)], dim=-1)

    orig_post_act_mean_individual: Dict = dict()
    dpo_post_act_mean_individual: Dict = dict()

    # Calculate average post activation for individual neuron
    for layer, idx in neuron_list:
        orig_post_act_mean_individual[(layer, idx)] = np.mean(
            orig_post_acts[(layer, idx)]
        )

        dpo_post_act_mean_individual[(layer, idx)] = np.mean(
            dpo_post_acts[(layer, idx)]
        )

    orig_post_act_mean_group = np.mean(list(orig_post_act_mean_individual.values()))
    dpo_post_act_mean_group = np.mean(list(dpo_post_act_mean_individual.values()))

    info: Dict[str, Dict[Tuple[int, int], List[float]]] = dict(
        orig=orig_post_act_mean_individual, dpo=dpo_post_act_mean_individual
    )

    return orig_post_act_mean_group, dpo_post_act_mean_group, info
