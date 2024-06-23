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
parser.add_argument("--model_id", type=str, help="Model ID", default="ai-forever/mGPT")
parser.add_argument("--aligned_text_dir", type=str, default="assets/translated_pairwise_data")
parser.add_argument("--cache_retrieval_dir", type=str, default="xg/retrieval/retrieval_cache")
args = parser.parse_args()

model_id = args.model_id.split("/")[-1] + "_" if args.model_id else ""
ALIGNED_TEXT_DIR = args.aligned_text_dir
CACHE_DIR = args.cache_retrieval_dir

IDXS = [x for x in range(0, 201, 50)] 
print("IDXS:", IDXS)

en_sentence_embeddings_all_layers_LBM = []
cn_sentence_embeddings_all_layers_LBM = []
for i, begin in enumerate(IDXS[:-1]):
    end = IDXS[i + 1]

    _en_embeddings = torch.load(
        osp.join(
            CACHE_DIR,
            f"{model_id}en_sentence_embeddings_all_layers_LBM_{begin}_{end}.pt",
        )
    )
    _cn_embeddings = torch.load(
        osp.join(
            CACHE_DIR,
            f"{model_id}{args.lang2}_sentence_embeddings_all_layers_LBM_{begin}_{end}.pt",
        )
    )

    if not en_sentence_embeddings_all_layers_LBM:
        en_sentence_embeddings_all_layers_LBM = _en_embeddings
    else:
        for j in range(len(en_sentence_embeddings_all_layers_LBM)):
            en_sentence_embeddings_all_layers_LBM[j] = torch.cat(
                [en_sentence_embeddings_all_layers_LBM[j], _en_embeddings[j]], dim=0
            )

    if not cn_sentence_embeddings_all_layers_LBM:
        cn_sentence_embeddings_all_layers_LBM = _cn_embeddings
    else:
        for j in range(len(cn_sentence_embeddings_all_layers_LBM)):
            cn_sentence_embeddings_all_layers_LBM[j] = torch.cat(
                [cn_sentence_embeddings_all_layers_LBM[j], _cn_embeddings[j]], dim=0
            )


def avg_se_L(se1_all_layers_LBM, se2_all_layers_LBM, n_layer=None):
    if n_layer is None:
        n_layer = len(se1_all_layers_LBM)

    avg_dist_L = []
    for i in range(n_layer):
        se1_BM = se1_all_layers_LBM[i]
        se2_BM = se2_all_layers_LBM[i]

        # Calculate Euclidean distance for each pair of embeddings in the batch
        distances_B = torch.nn.functional.cosine_similarity(se1_BM, se2_BM, dim=1)

        # Calculate the average distance for the current layer
        avg_distance = (
            distances_B.mean().item()
        )  # Convert to Python float with `.item()`
        avg_dist_L.append(avg_distance)

    return avg_dist_L


avg_se = avg_se_L(
    en_sentence_embeddings_all_layers_LBM, cn_sentence_embeddings_all_layers_LBM
)

print("Retrieval acc:", np.mean(avg_se))
