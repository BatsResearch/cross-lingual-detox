from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from xg.probe import Discriminator


def get_value_vector_projections(
    model_name: str, columns: List[Tuple[int, int]], topk: int
) -> Dict[Tuple[int, int], List[str]]:
    """
    Get topk projections in vocab space of the given set of value vectors.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unembed = model.lm_head.weight
    interps = {}

    for layer_id, neuron_id in tqdm(columns):
        down_column = model.transformer.h[layer_id].mlp.c_proj.weight[neuron_id]

        assert unembed.size(1) == down_column.size(0)

        projection = unembed @ down_column
        _, sorted_indices = torch.sort(projection, descending=True)

        interp = []
        for ind in sorted_indices[:topk]:
            interp.append(tokenizer.decode(ind))
        interps[(layer_id, neuron_id)] = interp
    return interps


def intervene_activations(
    orig_mlp_vector: torch.Tensor,
    hook,
    offset: float,
    neuron_group: List[Tuple[int, int]],
    debug_func: Callable = None,
) -> torch.Tensor:
    """
    Add offset to a group of neurons during forward pass.
    """

    current_layer = hook.layer()
    neuron_at_current_layer = [
        nid for (layer, nid) in neuron_group if layer == current_layer
    ]
    if not neuron_at_current_layer:
        return orig_mlp_vector

    for nid in neuron_at_current_layer:
        orig_mlp_vector[:, :, nid] += offset

    if debug_func is not None:
        debug_func()
    return orig_mlp_vector


def get_top_toxic_value_vectors(
    probe_path: str, model_name: str, topk: int
) -> List[Tuple[int, int]]:
    """Find the columns in the down projection that have the highest cosine similarities with the toxic probe"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    discreiminator = Discriminator.from_cache(
        pretrained_model=model_name, probe_path_or_module=probe_path
    )

    llm = discreiminator.pretrained_model
    probe = discreiminator.probe
    unembed = llm.lm_head
    unembed_weight = unembed.weight
    W_toxic = probe.weight.squeeze(0)
    W_toxic.shape

    stacked_params = []
    for name, param in llm.named_parameters():
        if "mlp.c_proj.weight" in name:
            stacked_params.append(param)
        elif "mlp.dense_4h_to_h.weight" in name:
            stacked_params.append(param.T)
    if not stacked_params:
        raise KeyError("Could not fetch value vectors with predefined keys")

    big_matrix = torch.cat(stacked_params, dim=0)
    unembed_weight = unembed_weight.to(device)
    big_matrix = big_matrix.to(device)
    W_toxic = W_toxic.to(device)
    cosine_similarities = F.cosine_similarity(big_matrix, W_toxic)
    _, top_ind = torch.topk(cosine_similarities, topk, largest=True)

    toxic_columns = []
    for global_id in top_ind:
        layer_id = global_id // 8192  # change if not using mGPT
        neuron_id = global_id % 8192

        toxic_columns.append((layer_id.item(), neuron_id.item()))
    return toxic_columns
