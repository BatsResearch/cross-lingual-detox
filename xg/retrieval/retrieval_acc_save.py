import argparse
import os.path as osp
import pathlib

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--lang1", type=str, help="Language 1", default="en")
parser.add_argument("--lang2", type=str, help="Language 2", default="zh")
parser.add_argument("--begin", type=int, help="Begin index", default=0)
parser.add_argument("--end", type=int, help="End index", default=25)
parser.add_argument("--model_name", type=str, help="Model name", default="ai-forever/mGPT")
parser.add_argument("--aligned_text_dir", type=str, default="assets/translated_pairwise_data")
parser.add_argument("--cache_retrieval_dir", type=str, default="xg/retrieval/retrieval_cache")
args = parser.parse_args()

ALIGNED_TEXT_DIR = args.aligned_text_dir
CACHE_DIR = args.cache_retrieval_dir
if not osp.exists(CACHE_DIR):
    pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

print("Saving sentence embeddings to:", CACHE_DIR)

model = AutoModelForCausalLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

tokenizer.padding_side = "right"
if "llama" in args.model_name.lower():
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 
model.to("cuda")


en_json = pd.read_json(
    f"{ALIGNED_TEXT_DIR}/dpo_toxicity_pairwise_en_200.jsonl",
    lines=True,
)
cn_json = pd.read_json(
    f"{ALIGNED_TEXT_DIR}/dpo_toxicity_pairwise_{args.lang2}_200.jsonl",
    lines=True,
)

en_text = en_json["Prompt"].tolist()
cn_text = cn_json["Prompt"].tolist()

BEGIN = args.begin
END = args.end

en_text = en_text[BEGIN:END]
cn_text = cn_text[BEGIN:END]

torch.set_grad_enabled(False)

en_inputs = tokenizer(en_text, return_tensors="pt", padding=True).to("cuda")
en_outputs = model(**en_inputs, output_hidden_states=True)

cn_inputs = tokenizer(cn_text, return_tensors="pt", padding=True).to("cuda")
cn_outputs = model(**cn_inputs, output_hidden_states=True)

en_hidden_states = en_outputs.hidden_states
en_attn_mask = en_inputs.attention_mask


cn_hidden_states = cn_outputs.hidden_states
cn_attn_mask = cn_inputs.attention_mask

assert len(en_hidden_states) == len(cn_hidden_states)


def sentence_embedding_BSM_to_BM(
    hidden_states_BSM: torch.Tensor, attn_mask_BS: torch.Tensor
) -> torch.Tensor:
    """Return sentence embedding average over all legal tokens"""

    assert len(hidden_states_BSM.shape) == 3, len(hidden_states_BSM.shape)
    assert len(attn_mask_BS.shape) == 2, len(attn_mask_BS.shape)
    assert hidden_states_BSM.size(0) == attn_mask_BS.size(0) and hidden_states_BSM.size(
        1
    ) == attn_mask_BS.size(1), f"{hidden_states_BSM.shape = } {attn_mask_BS.shape = }"

    # zero out hidden states for token masked out by attn mask
    attn_mask_BSM = attn_mask_BS.unsqueeze(dim=-1).expand_as(hidden_states_BSM)
    # print(f"{attn_mask_BSM = }")
    # element-wise mult
    masked_hidden_states_BSM = attn_mask_BSM * hidden_states_BSM
    # print(f"{masked_hidden_states_BSM = }")

    sum_hidden_states_BM = masked_hidden_states_BSM.sum(dim=1)
    # print(f"{sum_hidden_states_BM = }")
    token_counts_BS = attn_mask_BS.sum(
        dim=1, keepdim=True
    )  # Keeping dimension for broadcasting
    avg_hidden_states_BM = sum_hidden_states_BM / token_counts_BS
    return avg_hidden_states_BM


cn_sentence_embeddings_all_layers_LBM = []
en_sentence_embeddings_all_layers_LBM = []


for i in range(len(en_hidden_states)):
    en_hidden_states_BSM = en_hidden_states[i]
    cn_hidden_states_BSM = cn_hidden_states[i]

    en_sentence_embeddings_BM = sentence_embedding_BSM_to_BM(
        en_hidden_states_BSM, en_attn_mask
    )
    cn_sentence_embeddings_BM = sentence_embedding_BSM_to_BM(
        cn_hidden_states_BSM, cn_attn_mask
    )

    en_sentence_embeddings_all_layers_LBM.append(en_sentence_embeddings_BM)
    cn_sentence_embeddings_all_layers_LBM.append(cn_sentence_embeddings_BM)

torch.save(
    en_sentence_embeddings_all_layers_LBM,
    osp.join(
        CACHE_DIR,
        f"{args.model_name.split('/')[-1]}_{args.lang1}_sentence_embeddings_all_layers_LBM_{BEGIN}_{END}.pt",
    ),
)
torch.save(
    cn_sentence_embeddings_all_layers_LBM,
    osp.join(
        CACHE_DIR,
        f"{args.model_name.split('/')[-1]}_{args.lang2}_sentence_embeddings_all_layers_LBM_{BEGIN}_{END}.pt",
    ),
)