import os.path as osp
import pickle

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from xg import ASSET_DIR, DATASET_DIR
from xg.collect_neuron_activations import collect_neuron_activation_gpt2

actual_36 = [
    (13, 2337),
    (14, 6878),
    (14, 5723),
    (9, 6517),
    (12, 6538),
    (12, 6639),
    (3, 5794),
    (13, 3368),
    (1, 2583),
    (13, 7176),
    (23, 5888),
    (8, 7612),
    (11, 7033),
    (11, 4277),
    (18, 486),
    (16, 3531),
    (17, 520),
    (12, 3431),
    (5, 53),
    (10, 4641),
    (3, 3173),
    (7, 3971),
    (16, 4702),
    (17, 2392),
    (16, 4689),
    (21, 7155),
    (0, 7248),
    (17, 3530),
    (23, 2675),
    (11, 3027),
    (10, 8010),
    (10, 2127),
    (15, 594),
    (10, 7751),
    (10, 4920),
    (14, 7052),
]


EXP_GROUP = actual_36
EXP_LANGS = [
    "ar",
    "cs",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "sv",
    "zh-hans",
]


def main():

    torch.set_grad_enabled(False)

    # TODO: Download the RTP-LX dataset
    dataset_path = osp.join(DATASET_DIR, ...)
    dataset_df = pd.read_json(dataset_path, lines=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load original model and fine-tuned model
    orig_model = HookedTransformer.from_pretrained("mGPT", device=device)
    dpo_model_hf = AutoModelForCausalLM.from_pretrained("jmodel/mGPT-reprod-final")
    dpo_model = HookedTransformer.from_pretrained(
        "mGPT", hf_model=dpo_model_hf, device=device
    )

    # Enable batch processing
    orig_model.tokenizer.padding_side = "left"
    orig_model.tokenizer.pad_token_id = orig_model.tokenizer.eos_token_id

    dpo_model.tokenizer.padding_side = "left"
    dpo_model.tokenizer.pad_token_id = dpo_model.tokenizer.eos_token_id

    pre_dpo_postacts = dict()
    post_dpo_postacts = dict()

    for lang in EXP_LANGS:
        dataset = dataset_df[lang].tolist()
        prompt_tokens = orig_model.to_tokens(dataset)
        print(f"{lang} dataset shape: {prompt_tokens.shape}")

        orig_avg_postact, dpo_avg_postact, info = collect_neuron_activation_gpt2(
            orig_model=orig_model,
            dpo_model=dpo_model,
            neuron_list=EXP_GROUP,
            promopt_tokens=prompt_tokens,
        )

        print(f"Pre DPO AVG: {orig_avg_postact}, Post DPO AVG: {dpo_avg_postact}")

        pre_dpo_postacts[lang] = orig_avg_postact
        post_dpo_postacts[lang] = dpo_avg_postact

    print(pre_dpo_postacts)
    print(post_dpo_postacts)
    print(info)

    result = dict(
        pre=pre_dpo_postacts,
        post=post_dpo_postacts,
    )
    save_fp = osp.join(ASSET_DIR, "neuron_activations.pkl")
    with open(save_fp, "wb") as file:
        pickle.dump(result, file)


if __name__ == "__main__":
    main()
