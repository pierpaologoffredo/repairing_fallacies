from torch import bfloat16
import transformers
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
from tqdm import tqdm
import pandas as pd

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='This is a simple argument parser example.')

# Add arguments
parser.add_argument('--no_ctx', action='store_true', help="Set this flag if you don't want to consider the context")

# Parse the arguments
parse_args = parser.parse_args()

access_token = open('./personal/huggingface_token.txt').read()

start_time = time.time()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set an environment variable to enable expandable CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Create a quantization configuration for 4-bit precision and bfloat16 computation
quantization_config = transformers.BitsAndBytesConfig(load_in_4bit = True,
                                                      bnb_4bit_compute_dtype = torch.bfloat16)

# Specify the model ID for the pre-trained Mixtral-8x7B-Instruct-v0.1 model
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load the pre-trained model using AutoModelForCausalLM with the specified configurations
if model_id == 'mistralai/Mistral-7B-Instruct-v0.2':
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code = True,  # Trust the remote code during model loading
        torch_dtype = bfloat16,  # Use bfloat16 precision for tensor operations
        device_map = 'cuda',  # Automatically map the model across available devices
        token=access_token
    )
elif model_id == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code = True,  # Trust the remote code during model loading
        torch_dtype = bfloat16,  # Use bfloat16 precision for tensor operations
        device_map = 'cuda',  # Automatically map the model across available devices
        quantization_config = quantization_config,  # Use the specified quantization configuration
        token=access_token
    )

# Load the tokenizer for the pre-trained model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=access_token)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create a text generation pipeline using the loaded model and tokenizer
generate_text = transformers.pipeline(
    model = model,  # The loaded model
    tokenizer = tokenizer,  # The loaded tokenizer
    return_full_text = False,  # Return only the generated text, not the full output
    task = "text-generation",  # Specify the task as text generation
    do_sample = True,  # Enable sampling for generating text
    temperature = 0.1,  # Controls the randomness of outputs (lower values are more deterministic)
    top_p = 0.15,  # Probability threshold for selecting tokens (nucleus sampling)
    top_k = 0,  # Number of top tokens to consider (0 relies on top_p)
    max_new_tokens = 100,  # Limits the number of generated tokens
    repetition_penalty = 1.1,  # Discourages repetitive outputs
    torch_dtype = torch.bfloat16  # Use bfloat16 precision for tensor operations
)

if parse_args.no_ctx:
    data_path = "./data/mistral_data/few_shot_mistral_no_ctx_lab.txt"
else:
    data_path = "./data/mistral_data/few_shot_mistral_lab.txt"

results = []

with open(data_path, "r", encoding='utf-8') as data:
    data_lines = data.readlines()
    for prompt in tqdm(data_lines):
        prompt_ = prompt
        model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to('cuda')
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=True, temperature=0.5)
        decoded = tokenizer.batch_decode(generated_ids)
        result_ = decoded[0]
        results.append([prompt_, result_])
        print(result_)

df_results = pd.DataFrame(results, columns=['prompt', 'results'])

path_res = "./fs_results/"
if parse_args.no_ctx:
    file_res = "{}_fs_results_no_ctx_lab.csv".format(model_id)
else:
    file_res = "{}_fs_results_lab.csv".format(model_id)

df_results.to_csv(os.path.join(path_res, file_res.split("/")[1]), index=False)
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.6f} seconds")