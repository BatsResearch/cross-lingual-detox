"""
Code taken from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py
Blog: https://huggingface.co/blog/dpo-trl
"""

import json
# 0. imports
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          EarlyStoppingCallback, HfArgumentParser,
                          TrainingArguments, set_seed)
from trl import DPOTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.

    Note: Following Table 5 in https://arxiv.org/pdf/2401.01967
    """

    # data parameters
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    data_dir: Optional[str] = field(
        default="/users/zyong2/data/zyong2/m_mech_toxic/data/external/dpo_toxic_artifacts/toxicity_pairwise/"
    )
    train_perc: Optional[float] = field(
        default=1.0, metadata={"help": "the percentage of training data"}
    )

    # training parameters
    # follow https://github.com/yongzx/dpo_toxic/blob/main/toxicity/train_dpo/config/config.yaml
    model_name_or_path: Optional[str] = field(
        default="ai-forever/mGPT",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(
        default=1e-5, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(
        default=150, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.05, metadata={"help": "the weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="rmsprop", metadata={"help": "the optimizer type"}
    )
    max_grad_norm: Optional[float] = field(
        default=10.0, metadata={"help": "the maximum gradient norm"}
    )

    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "eval batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use reentrant for gradient checkpointing"},
    )

    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "whether to use lora"}
    )

    ## lora-specific arguments
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha (scaling) parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(
        default=64, metadata={"help": "the lora r (rank) parameter"}
    )

    max_prompt_length: Optional[int] = field(
        default=64, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=256, metadata={"help": "the maximum sequence length"}
    )
    # max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(
        default=5, metadata={"help": "number of training epochs"}
    )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "the logging frequency"}
    )
    save_steps: Optional[int] = field(
        default=25, metadata={"help": "the saving frequency"}
    )
    eval_steps: Optional[int] = field(
        default=25, metadata={"help": "the evaluation frequency"}
    )

    output_dir: Optional[str] = field(
        default="/users/zyong2/data/zyong2/m_mech_toxic/data/processed/002-eval-mgpt/mgpt-dpo",
        metadata={"help": "the output directory"},
    )
    log_freq: Optional[int] = field(
        default=1, metadata={"help": "the logging frequency"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "whether to load the model in 4bit"}
    )
    model_dtype: Optional[str] = field(
        default="float",
        metadata={
            "help": "model_dtype[float16, bfloat16, float] for loading."
        },  # float16 causes NaN reward
    )

    # instrumentation
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    wandb_run_name: Optional[str] = field(
        default="dpo", metadata={"help": "the wandb run name"}
    )


def get_toxicity_pairwise_data(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """

    data_dir = pathlib.Path(data_dir)
    data = list()
    for jsonl_fn in data_dir.glob("*.jsonl"):
        with open(jsonl_fn, "r") as rf:
            for line in rf:
                line = json.loads(line.strip())
                if "prompt_input_ids" in line:
                    del line["prompt_input_ids"]

                if "gold_input_ids" in line:
                    del line["gold_input_ids"]

                if "pert_input_ids" in line:
                    del line["unpert_gen_token_ids"]

                if "pert_gen_toks" in line:
                    del line["pert_gen_toks"]

                data.append(line)

    # filter out train_perc of data
    print(f"Total number of samples: {len(data)}")
    data = data[: int(len(data) * script_args.train_perc)]
    print(
        f"Total number of samples after filtering ({script_args.train_perc}%): {len(data)}"
    )

    # prompt_text: prompt
    # unpert_gen_text: good response (chosen)
    # pert_gen_text: bad response (rejected)

    dataset = Dataset.from_pandas(pd.DataFrame(data=data))
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": samples["prompt_text"],
            "chosen": samples["unpert_gen_text"],
            "rejected": samples["pert_gen_text"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=script_args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_toxicity_pairwise_data(
        data_dir=script_args.data_dir, sanity_check=script_args.sanity_check
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    # 3. Load evaluation dataset
    # eval_dataset = get_toxicity_pairwise_data(data_dir=script_args.data_dir, sanity_check=True)
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )
    train_dataset = train_dataset.train_test_split(
        test_size=0.05, seed=script_args.seed
    )
    eval_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]
    print("Example training sample:")
    print(train_dataset[0])
    print(train_dataset[1])
    print(train_dataset[2])
    print("=" * 50)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        max_grad_norm=script_args.max_grad_norm,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=False,  # false because V100 doesn't support bf16
        fp16=True,
        remove_unused_columns=False,
        run_name=script_args.wandb_run_name,
        gradient_checkpointing_kwargs=dict(
            use_reentrant=script_args.gradient_checkpointing_use_reentrant
        ),
        seed=script_args.seed,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,  # final_checkpoint saves the best model (lowest validation loss)
        save_total_limit=5,
    )

    # 5. initialize the PEFT config

    peft_target_modules = None
    if script_args.use_lora:
        if (
            "llama" in script_args.model_name_or_path.lower()
            or "mala" in script_args.model_name_or_path.lower()
        ):
            peft_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
                "lm_head",
            ]
        elif "bloom" in script_args.model_name_or_path.lower():
            peft_target_modules = [
                "word_embeddings",
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
        elif "aya" in script_args.model_name_or_path.lower():
            peft_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
                "lm_head",
            ]

        assert (
            peft_target_modules is not None
        ), "Please specify the target modules for PEFT"
        print(">>>> Loaded LoRA target modules: ", peft_target_modules)

        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=peft_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        print(">>>> Full model finetuning")
        peft_config = None

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    final_output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(final_output_dir)  # save best model
    tokenizer.save_pretrained(final_output_dir)
