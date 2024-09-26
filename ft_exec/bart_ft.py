import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import BartForConditionalGeneration
from datasets import load_dataset, load_from_disk
import numpy as np
import nltk
nltk.download('punkt')
from datasets import Dataset, DatasetDict
import pandas as pd
import os 
import wandb
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

max_input = 512
max_target = 128
batch_size = 3
model_checkpoints = "facebook/bart-large"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)

def process_dataset_(model: str):
    if model == 'bart':
        if parse_args.no_ctx:
            train = pd.read_csv(f'./data/{model}_data/train_{model}_no_ctx.csv')
            val = pd.read_csv(f'./data/{model}_data/val_{model}_no_ctx.csv')
            
            train_data = [{"starting": ctx, 'fixed': fix} for ctx, fix in zip(train.starting.tolist(), train.fixed.tolist())]
            val_data = [{"starting": ctx, 'fixed': fix} for ctx, fix in zip(val.starting.tolist(), val.fixed.tolist())]
        else:
            train = pd.read_csv(f'./data/{model}_data/train_{model}.csv')
            val = pd.read_csv(f'./data/{model}_data/val_{model}.csv')
            
            train_data = [{"original": ctx, 'fixed': fix} for ctx, fix in zip(train.Context.tolist(), train.fixed_context.tolist())]
            val_data = [{"original": ctx, 'fixed': fix} for ctx, fix in zip(val.Context.tolist(), val.fixed_context.tolist())]
    else:
        train_lines = open(f'./data/{model}_data/train_{model}.txt').readlines()
        val_lines = open(f'./data/{model}_data/val_{model}.txt').readlines()

        train_data = [{"prompt": line} for line in train_lines]
        val_data = [{"prompt": line} for line in val_lines]

    # Create the Dataset objects
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Create the DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
    })

    return dataset_dict
    
bart_data = process_dataset_('bart')

def preprocess_data(data_to_process, no_ctx: bool=None):
  
    #get all the prompt
    if parse_args.no_ctx:
        inputs = [original for original in data_to_process['starting']]
    else:
        inputs = [original for original in data_to_process['original']]

    #tokenize the prompt
    model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)

    #tokenize the fixed arguments
    with tokenizer.as_target_tokenizer():
        if parse_args.no_ctx:
            targets = tokenizer(data_to_process['fixed'], max_length=max_target, padding='max_length', truncation=True)
        else:
            targets = tokenizer(data_to_process['fixed_context'], max_length=max_target, padding='max_length', truncation=True)
            
    model_inputs['labels'] = targets['input_ids']
    return model_inputs

tokenized_data = bart_data.map(preprocess_data, batched = True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)

args = Seq2SeqTrainingArguments(
    output_dir='./ft_models/bart-ft-no-ctx', #save directory
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size= 2,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    eval_accumulation_steps=3,
    fp16=True, #available only with CUDA,
    report_to="wandb",
    run_name="bart_ft",  # name of the W&B run (optional)
    )

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

if parse_args.no_ctx:
    test = pd.read_csv(f'./data/bart_data/test_bart_no_ctx.csv')
    save_path = "./ft_models/bart_finetuned_no_ctx"
else:
    save_path = "./ft_models/bart_finetuned"
    test = pd.read_csv(f'./data/bart_data/test_bart.csv')

trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

ft_model = BartForConditionalGeneration.from_pretrained(save_path)
ft_tokenizer = AutoTokenizer.from_pretrained(save_path)

results = []
for i, row in test.iterrows():
    if parse_args.no_ctx:
        prompt = row['starting']
    else:
        prompt = row['Context']
    inputs = ft_tokenizer([prompt], max_length=1024, return_tensors="pt")
    gen_txt_ids = ft_model.generate(inputs['input_ids'], num_beams=2, min_length=0, max_length=128)
    gen_txt = tokenizer.batch_decode(gen_txt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    results.append([prompt, gen_txt])
    
df_results = pd.DataFrame(results, columns=['prompt', 'results'])

path_res = "./ft_results/"
if parse_args.no_ctx:
    file_res = "{}_ft_results_no_ctx.csv".format(model_checkpoints)
else:
    file_res = "{}_ft_results.csv".format(model_checkpoints)

df_results.to_csv(os.path.join(path_res, file_res.split("/")[1]), index=False)