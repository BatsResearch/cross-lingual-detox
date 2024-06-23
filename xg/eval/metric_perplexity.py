#####
"""
https://github.com/for-ai/goodtriever/blob/81aa697841c56993b935167484484253f704629a/scripts/utils/evaluation_metrics.py#L56
"""
import argparse
import gc
import json
import pathlib

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_outputs_folder", type=str)
parser.add_argument("--eval_model_id", type=str, default="google/mt5-xl")

args = parser.parse_args()
print("Arguments")
print(vars(args))

args.model_outputs_folder = pathlib.Path(args.model_outputs_folder)


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.eval_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(args.eval_model_id).to(device)

langs = []
medians = []
q1s = []
q3s = []
means = []
for prompt_file in args.model_outputs_folder.glob("*.json"):
    ppl_file_name = args.model_outputs_folder / (prompt_file.stem + "_ppl.pt")
    if ppl_file_name.exists():
        print("Skipping", ppl_file_name, "as it already exists")
        continue
    with open(prompt_file, "r") as file_p:
        lang = prompt_file.stem.split("_")[-1]
        print(lang)

        ppls = []
        for line_i, line in tqdm(enumerate(file_p)):
            prompt = json.loads(line.strip())["prompt"]
            generated_text = json.loads(line.strip())["generated_text"]
            if not generated_text or not generated_text.strip():
                continue

            full_text = prompt + generated_text
            full_input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (
                full_input_ids.shape[1] - 1
            )

            prompt_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (
                prompt_input_ids.shape[1] - 1
            )

            loss = (full_loss - prompt_loss) / (
                full_input_ids.shape[1] - prompt_input_ids.shape[1]
            )

            ppl = np.exp(loss.item())

            if ppl < 1e4:  # for sanity
                ppls.append(ppl)

        median = np.percentile(ppls, 50, interpolation="midpoint")
        q1 = np.percentile(ppls, 25, interpolation="midpoint")
        q3 = np.percentile(ppls, 75, interpolation="midpoint")
        mean = np.mean(ppls)
        print("Median:", median, f"({q1}, {q3})")
        print("Mean:", np.mean(ppls))
        print("Std:", np.std(ppls))

        langs.append(lang)
        medians.append(round(median, 2))
        q1s.append(round(q1, 2))
        q3s.append(round(q3, 2))
        means.append(round(mean, 2))

        ppl_file_name = args.model_outputs_folder / (prompt_file.stem + "_ppl.pt")
        print("Saving perplexities array to", ppl_file_name)
        torch.save(ppls, ppl_file_name)

print("Langs:", langs)
print("Medians:", medians)
print("Q1s:", q1s)
print("Q3s:", q3s)
# print("Means:", means)