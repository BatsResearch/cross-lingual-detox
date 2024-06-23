import argparse
import json
import pathlib

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def compute_diversity(generations, tokenizer):
    # calculate diversity across generations for every prompt
    dist1, dist2, dist3 = [], [], []

    total_tokens = 0
    unigrams, bigrams, trigrams = set(), set(), set()
    for i, gen in enumerate(generations):

        o = tokenizer(gen)["input_ids"]
        ow = [tokenizer.decode(x, skip_special_tokens=True) for x in o]
        ow = [x for x in ow if x]

        # print(gen, ow, len(ow))

        unigrams.update(ow)
        for i in range(len(ow) - 1):
            bigrams.add(ow[i] + "_" + ow[i + 1])
        for i in range(len(ow) - 2):
            trigrams.add(ow[i] + "_" + ow[i + 1] + "_" + ow[i + 2])

        total_tokens += len(ow)

        # print(unigrams, bigrams, trigrams)
        # print(len(unigrams), len(bigrams), len(trigrams))
    print(len(generations), generations[:2])
    dist1 = len(unigrams) / total_tokens
    dist2 = len(bigrams) / total_tokens
    dist3 = len(trigrams) / total_tokens
    return dist1, dist2, dist3


parser = argparse.ArgumentParser()
parser.add_argument("--model_outputs_folder", type=str, required=True)
parser.add_argument("--tokenizer", type=str, default="google/mt5-xl")
parser.add_argument("--sample", type=int, default=25)
args = parser.parse_args()

### show arguments for debugging
print("Arguments:")
print(vars(args))

model_outputs_folder = pathlib.Path(args.model_outputs_folder)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

langs = []
dist1s = []
dist1s_std = []
dist2s = []
dist2s_std = []
dist3s = []
dist3s_std = []
SAMPLE = args.sample
for file in tqdm(model_outputs_folder.glob("*.json"), desc="Processing files"):
    lang = file.stem.split("_")[-1]
    with open(file, "r") as rf:
        all_generations = [json.loads(line)["generated_text"] for line in rf]

    all_dist1, all_dist2, all_dist3 = [], [], []
    # diversity of k generations per prompt
    for i in range(0, len(all_generations), SAMPLE):
        generations = all_generations[i : i + SAMPLE]
        generations = [gen for gen in generations if gen]
        if len(generations) == 0:
            continue
        dist1, dist2, dist3 = compute_diversity(
            generations, tokenizer
        )  # compute diversity of k continuations per prompt
        all_dist1.append(dist1)
        all_dist2.append(dist2)
        all_dist3.append(dist3)

    langs.append(lang)
    dist1s.append(round(np.mean(all_dist1), 3))
    dist1s_std.append(round(np.std(all_dist1), 3))
    dist2s.append(round(np.mean(all_dist2), 3))
    dist2s_std.append(round(np.std(all_dist2), 3))
    dist3s.append(round(np.mean(all_dist3), 3))
    dist3s_std.append(round(np.std(all_dist3), 3))

    print(f"Language: {lang}")
    print(f"Unigram diversity: {round(np.mean(all_dist1), 3)}")
    print(f"Bigram diversity: {round(np.mean(all_dist2), 3)}")
    print(f"Trigram diversity: {round(np.mean(all_dist3), 3)}")

print("Languages:", langs)
print("Unigram diversity:", dist1s)
print("Unigram diversity std:", dist1s_std)
print("Bigram diversity:", dist2s)
print("Bigram diversity std:", dist2s_std)
print("Trigram diversity:", dist3s)
print("Trigram diversity std:", dist3s_std)
