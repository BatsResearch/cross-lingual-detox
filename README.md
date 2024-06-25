# Preference Tuning For Toxicity Mitigation Generalizes Across Languages

<p align="center">
    <a href="https://arxiv.org/abs/2406.16235"><img src="https://badgen.net/static/arxiv/2406.16235/blue" /></a>
    <a href="https://x.com/yong_zhengxin/status/xxxxxxxxxx"><img src="https://badgen.net/static/Twitter/summary/blue?icon=twitter" /></a>
</p>

🔥 **Cross-lingual safety generalization**: This is the first work to demonstrate preference tuning for toxicity mitigation can generalize cross-lingually in a zero-shot manner. We evaluated on 17 different languages and different LLMs (such as BLOOM, Llama3, and Aya-23), all of which shows cross-lingual detoxification after English DPO preference tuning.

🔍 **Mechanistic findings**: We show that the *dual multilinguality* of toxic vectors (in MLP layers) explains the cross-lingual generalization. We find that the
toxic vectors in MLPs encode multilingual toxic concepts, and we can control the output toxicity level by controlling the activation levels of those vectors. We then show that English DPO reduces activation levels of toxic vectors across languages.

---

## Table of Contents

- [Setup](#setup)
- [DPO Preference Tuning](#dpo-preference-tuning)
  - [0. Download Training and Evaluation Data](#0-download-training-and-evaluation-data)
  - [1. Training](#1-training)
  - [2. Generation](#2-generation)
  - [3. Evaluation](#3-evaluation)
- [Interpretability Experiments](#interpretability-experiments)
  - [0. Download Jigsaw toxic comments dataset](#0-download-jigsaw-toxic-comments-dataset)
  - [1. Training Probe](#1-probe-training)
  - [2. Interpreting Value Vectors](#2-analyze-the-value-vectors-of-model)
  - [3. Causal Intervention](#3-causal-intervention)
  - [4. Compare Neuron Activations](#4-analyze-neuron-activation-before-and-after-dpo)
- [Bilingual Sentence Retrieval](#bilingual-sentence-retrieval)
- [Bibtex](#bibtex)

---

### Setup

1. Create a conda environment with python version 3.11
```bash
conda create --name xgdetox python=3.11
conda activate xgdetox
```

2. Install poetry and other dependencies with poetry. (Make sure you are at project's root directory, where pyproject.toml locates.)
```bash
pip install poetry 
poetry install 
```

---
### DPO Preference Tuning

#### 0. Download Training and Evaluation Data

- **Training (Toxicity Pairwise Data)**: Download the `toxicity_pairwise.zip` data from [here](https://drive.google.com/drive/folders/1baArqcjIc2Q4OllLVUz1hp3p3XxmdteK?usp=drive_link) (Source: [Mechanistically Understanding DPO: Toxicity](https://github.com/ajyl/dpo_toxic?tab=readme-ov-file)).

- **Evaluation (RTP-LX)**: Follow [instructions from Microsoft](https://github.com/microsoft/RTP-LX/tree/main) to download the dataset of RTP-LX input prompts. It will contain files of `RTP-LX/RTP_LX_{language}.json`. Our repo and experiments use the dataset released in Apr'24 (May'24 works too).

#### 1. Training

To perform DPO preference tuning (with or without LoRA), simply follow the following code example:

```bash
python3 xg/training/dpo.py \
    --data_dir /path/to/toxicity_pairwise/ \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir /path/to/save/model_ckpt \
    --per_device_train_batch_size 4 \
    --wandb_run_name your_wandb_runname \
    --use_lora  # remove this line if you want to do full model finetuning 
```

**After DPO training, you can directly use the model checkpoint from `/path/to/save/model_ckpt/final_checkpoint/`.**

However, because parameter-efficient training with LoRA adapters save the adapters, use the following code to merge the LoRA adapters and save the model weight. This helps with vLLM library for generation stage (at the time we design the code, there are bugs with loading LoRA weights so it is more straightforward to pass the merged model instead of base model + lora weights).

```bash
python3 xg/training/merge_peft.py \
    --base_model_name meta-llama/Llama-2-7b-hf \
    --lora_adapter /path/to/save/model_ckpt/final_checkpoint \
    --output_dir /path/to/save/merge_final_checkpoint
```

We have uploaded our trained models to HuggingFace Hub:
- [mGPT](https://huggingface.co/BatsResearch/mGPT-detox)
- [Bloom-1b7](https://huggingface.co/BatsResearch/bloom-1b7-detox)
- [Bloom-7b1](https://huggingface.co/BatsResearch/bloom-7b1-detox-qlora)
- [Llama2](https://huggingface.co/BatsResearch/llama2-7b-detox-qlora)
- [Llama3](https://huggingface.co/BatsResearch/llama3-8b-detox-qlora)
- [Aya-23](https://huggingface.co/BatsResearch/aya-23-8b-detox-qlora)

#### 2. Generation

We use the [vLLM library](https://docs.vllm.ai/en/latest/index.html) to obtain model continuations. We recommend user follow their [installation instruction](https://docs.vllm.ai/en/latest/getting_started/installation.html) before running the following generation code. Our code saves the vLLM generations as `/path/to/save/outputs/{MODEL_NAME}/output-rtp_lx_{LANG}.json`

```bash
PROMPT_FILE=/path/to/RTP-LX/RTP_LX_ZH-Hans.json # you can change the language to other languages than ZH-Hans
python3 xg/generate/vllm_script_sample.py \
    --prompt_file $PROMPT_FILE \
    --model /path/to/save/merge_final_checkpoint \  # or /path/to/save/model_ckpt/final_checkpoint (if you do full finetuning)
    --output_dir /path/to/save/outputs
```

#### 3. Evaluation

- **Toxicity**: First run the `xg/eval/perspective_api_eval.py` to save the toxicity scores from Perspective API. Then run `xg/eval/metric_toxicity.py` to aggregate the scores.

- **Fluency**: Run the `xg/eval/metric_perplexity.py` script to compute median conditional perplexity with the `mT5-xl` model. It will also save the array of all perplexity scores. 

- **Diversity**: Run the `xg/eval/metric_diversity.py` script.
```bash
MODEL_OUTPUTS_FOLDER=... # vllm generations folder (/path/to/save/outputs/{MODEL_NAME})

############### toxicity ###############
# call Perspective API
LANGS=( ar cs de en es fr hi id it ja ko nl pl pt ru sv zh-hans )
for LANG in "${LANGS[@]}"
do
    echo "Processing $LANG"
    python3 xg/eval/perspective_api_eval.py \
        --api_key ... \  # YOUR API KEY
        --datapath "${MODEL_OUTPUTS_FOLDER}/output-rtp_lx_${LANG}.json" \
        --output_folder "${MODEL_OUTPUTS_FOLDER}/perspective_api_eval/" \
        --language $LANG
done

# aggregate toxicity scores
PERSPECTIVE_OUTPUTS_FOLDER=${MODEL_OUTPUTS_FOLDER}/perspective_api_eval
python3 xg/eval/metric_toxicity.py \
    --perspective_outputs_folder $PERSPECTIVE_OUTPUTS_FOLDER

############### fluency ###############
python3 xg/eval/metric_perplexity.py \
    --model_outputs_folder $MODEL_OUTPUTS_FOLDER

############### diversity ###############
python3 xg/eval/metric_diversity.py \
    --model_outputs_folder $MODEL_OUTPUTS_FOLDER
```

---

### Interpretability Experiments
#### 0. Download Jigsaw Toxic Comments Dataset
Download the Jigsaw dataset from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). 

#### 1. Probe training
To train a linear probe for binary toxic classification, follow these steps:

- Replace the train_fp variable with the path to the train split of the Jigsaw dataset.
- Run the provided script.

All hyperparameters are pre-configured in the script file.
```bash
python scripts/run_train_probe.py
```

#### 2. Analyze the value vectors of model. 
We first identify the potential sources of toxicity by selecting the top 100 value vectors based on their cosine similarities with the probe vector. Then, we collect the corresponding neuron activations averaged across the next 20 tokens generated from the English RTP-LX prompt. The value vectors are retained if their corresponding neuron activations are positive during the forward pass. We found 36 value vectors meeting these criteria, and they are stored [here](assets/actual_sources_of_toxicity.pkl). We then project them onto the vocabulary space to interpret the tokens they promote when activated. More details can be found in [this notebook](<notebooks/Interpreting Value Vectors (Table 3, 7, and 8).ipynb>).

#### 3. Causal Intervention 
To better understand these sub-updates, we directly intervene in their corresponding and inspect the changes they induce. We provide a minimal experiment demonstrating how such interventions are conducted in [this notebook](<notebooks/Causal Intervention (Table 4).ipynb>). The same code can be used to quantitatively understand the effect of the changes we exert on the neuron activations across all prompts from differnt langauges.

#### 4. Analyze neuron activation before and after DPO 
[This script](scripts/run_collect_neuron_activations.py) can be used to collect neuron activations before and after preference tuning across different languages. We also provide the precomputed results [here](assets/neuron-activations.pkl). The reproduce Figure 3 in the paper, see [this notebook](<notebooks/Neuron Activation (Figure 3).ipynb>).

---

### Bilingual Sentence Retrieval

**Data**: Since that RTP-LX prompts are not aligned (see [Issue](https://github.com/microsoft/RTP-LX/issues/2)), we translate 200 prompts with Google Translate API so we have multiway parallel RTP-LX prompts. This is stored at `assets/translated_pairwise_data`.

We first use `xg/retrieval/retrieval_acc_save.py` to save the per-layer representations for parallel sentence pairs in English and `lang2` language. Then, we use `xg/retrieval/retrieval_acc_load.py` to load and calculate the bilingual sentence retrieval accuracy between English and `lang2`.

```bash
LANG2="ar"
for i in "0 50" "50 100" "100 150" "150 200" # process in batches to avoid OOM
do
    set -- $i # Convert the "tuple" into the param args $1 $2...
    python3 xg/retrieval/retrieval_acc_save.py \
        --lang2 $LANG2 \
        --begin $1 \
        --end $2 \
        --model_name "ai-forever/mGPT"
done

python3 xg/retrieval/retrieval_acc_load.py \
  --lang2 $LANG2
```

---
### Bibtex
```
@misc{li2024preference,
      title={Preference Tuning For Toxicity Mitigation Generalizes Across Languages}, 
      author={Xiaochen Li and Zheng-Xin Yong and Stephen H. Bach},
      year={2024},
      eprint={2406.16235},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
