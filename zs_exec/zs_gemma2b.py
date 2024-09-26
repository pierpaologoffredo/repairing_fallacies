from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import torch
import pandas as pd
import os
from tqdm import tqdm

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='This is a simple argument parser example.')

# Add arguments
parser.add_argument('--no_ctx', action='store_true', help="Set this flag if you don't want to consider the context")

# Parse the arguments
parse_args = parser.parse_args()


access_token = open('./personal/huggingface_token.txt').read()

modelName = "google/gemma-1.1-2b-it"

bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    modelName,
    device_map = "auto",
    quantization_config=bnbConfig,
    token=access_token
)

tokenizer = AutoTokenizer.from_pretrained(modelName, token=access_token)

if parse_args.no_ctx:
    data_path = "./data/gemma_data/zero_shot_gemma_no_ctx_lab.txt"
else:
    data_path = "./data/gemma_data/zero_shot_gemma_lab.txt"

results = []

with open(data_path, "r", encoding='utf-8') as data:
    data_lines = data.readlines()
    for prompt in tqdm(data_lines):
        prompt_ = prompt            
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
        outputs = model.generate(**inputs, max_length=512, num_return_sequences=1, temperature=0.5)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result_ = text
        results.append([prompt_, result_])
        # print(text)

df_results = pd.DataFrame(results, columns=['prompt', 'results'])

path_res = "./zs_results/"
if parse_args.no_ctx:
    file_res = "{}_zs_results_no_ctx_lab.csv".format(modelName)
else:
    file_res = "{}_zs_results_lab.csv".format(modelName)

df_results.to_csv(os.path.join(path_res, file_res.split("/")[1]), index=False)