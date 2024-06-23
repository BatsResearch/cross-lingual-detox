import argparse
import json
import pathlib

from tqdm import tqdm
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--prompt_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
###### commenting because do_sample=False (https://github.com/yongzx/dpo_toxic/blob/0ab76996b2fa5e725ad00985de9e284a1121c31e/toxicity/eval_interventions/generate_funcs.py#L43)
parser.add_argument("--N", type=int, default=25)
parser.add_argument("--temperature", type=float, default=0.9)  # see config on
parser.add_argument("--top_p", type=float, default=0.8)
########################
parser.add_argument(
    "--max_tokens", type=int, default=20
)  # currently following faithful reproduction https://arxiv.org/abs/2401.01967
parser.add_argument("--model", type=str, default="ai-forever/mGPT")
args = parser.parse_args()

### show arguments for debugging
print("Arguments:")
print(vars(args))

prompt_file = pathlib.Path(args.prompt_file)
model_name = args.model.split("/")[-1]
if "checkpoint" in model_name:
    model_name = args.model.split("/")[-2] + f"_{model_name}"
output_dir = pathlib.Path(args.output_dir) / f"{model_name}"
output_dir.mkdir(parents=True, exist_ok=True)
print("Output directory:", output_dir)

prompts = list()
with open(args.prompt_file) as rf:
    for line in rf:
        prompt = json.loads(line.strip())
        prompts.append(prompt["Prompt"])

print("Number of prompts:", len(prompts))
print("Example prompt:", prompts[0])

# Create a sampling params object.
sampling_params = SamplingParams(
    n=args.N,
    temperature=args.temperature,
    top_p=args.top_p,
    #  top_k=args.top_k,
    max_tokens=args.max_tokens,
)

# Create an LLM.
llm = LLM(model=args.model, swap_space=32)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

# Print the outputs.
write_file = output_dir / f"output-{prompt_file.stem.lower()}.json"
wf = open(write_file, "a+", buffering=1)
for output in tqdm(outputs):
    prompt = output.prompt

    for i in range(args.N):
        generated_text = output.outputs[i].text
        wf.write(
            json.dumps({"prompt": prompt, "generated_text": generated_text}) + "\n"
        )
