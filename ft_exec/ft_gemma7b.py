from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import torch 
from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTTrainer
import wandb
import os
import pandas as pd

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='This is a simple argument parser example.')

# Add arguments
parser.add_argument('--no_ctx', action='store_true', help="Set this flag if you don't want to consider the context")

# Parse the arguments
parse_args = parser.parse_args()

wandb_key = open('./personal/wandb_token.txt').read()
wandb.login(key=wandb_key)
os.environ["WANDB_PROJECT"] = "fine-tuning-unveiling"

base_model = "google/gemma-1.1-7b-it"

hf_token = open('./personal/huggingface_token.txt').read()

#Loading the dataset
def process_dataset(model: str):
    if parse_args.no_ctx:
        train_lines = open(f'./data/{model}_data/train_{model}_no_ctx_lab.txt').readlines()
        val_lines = open(f'./data/{model}_data/val_{model}_no_ctx_lab.txt').readlines()
    else:
        train_lines = open(f'./data/{model}_data/train_{model}_lab.txt').readlines()
        val_lines = open(f'./data/{model}_data/val_{model}_lab.txt').readlines()

    train_data = [{"prompt": line} for line in train_lines]
    val_data = [{"prompt": line} for line in val_lines]

    # Create the Dataset objects
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Create the DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_dataset
    })

    return dataset_dict

data = process_dataset('gemma')

train_data = data['train']

# Load base model(Gemma 7B-it)
bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnbConfig,
        device_map={"": 0},
        token=hf_token
)

model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj']
)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./ft_models/gemma7b",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    optim="paged_adamw_32bit",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    logging_strategy="steps",
    learning_rate=2e-4,
    load_best_model_at_end=True, 
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb",
    run_name="ft_gemma7b",  # name of the W&B run (optional)
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=data['val'],
    peft_config=peft_config,
    max_seq_length= 1024,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= True,
    
)

trainer.train()

if parse_args.no_ctx:
    new_model = "./ft_models/ft_gemma7b_no_ctx"
else:
    new_model = "./ft_models/ft_gemma7b"

# Clear the memory footprint
del model, trainer
torch.cuda.empty_cache()

### Fine-tuning LLMs with PEFT and LoRA
b_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    quantization_config=bnbConfig,
    device_map={"": 0},
    token=hf_token
)

ft_model = PeftModel.from_pretrained(b_model, new_model)
ft_model = ft_model.merge_and_unload()

tok = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True)

if parse_args.no_ctx:
    test_lines = open('./data/gemma_data/test_gemma_no_ctx_lab.txt').readlines()
else:
    test_lines = open('./data/gemma_data/test_gemma_lab.txt').readlines()

results = []
from tqdm import tqdm
for prompt in tqdm(test_lines):
    inputs2 = tok(prompt, return_tensors="pt").to({"": 0})
    outputs2 = ft_model.generate(**inputs2, max_new_tokens=512, num_return_sequences=1, num_beams=5, early_stopping=True, no_repeat_ngram_size=2, do_sample=True, temperature=0.5)
    txt2 = tok.decode(outputs2[0])
    
    results.append([prompt, txt2])
df_results = pd.DataFrame(results, columns=['prompt', 'results2'])

path_res = "./ft_results/"
if parse_args.no_ctx:
    file_res = "{}_ft_results_no_ctx_lab.csv".format(base_model)
else:
    file_res = "{}_ft_results_lab.csv".format(base_model)

df_results.to_csv(os.path.join(path_res, file_res.split("/")[1]), index=False)