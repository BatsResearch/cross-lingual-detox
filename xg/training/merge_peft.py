import argparse
import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_name", type=str, default="bigscience/bloom-7b1")
parser.add_argument("--lora_adapter", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

base_model_name = args.base_model_name
lora_adapter = pathlib.Path(args.lora_adapter)
save_to_dir = (
    pathlib.Path(args.output_dir)
    / base_model_name.split("/")[-1]
    / f"lora_{lora_adapter.parent.name}_{lora_adapter.stem}"
)
print("Saving model to directory: >>> ", save_to_dir)

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model_to_merge = PeftModel.from_pretrained(base_model, lora_adapter)
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(save_to_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(save_to_dir)
