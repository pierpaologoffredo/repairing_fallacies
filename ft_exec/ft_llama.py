from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
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
from huggingface_hub import login
import os
hf_token = open('./personal/huggingface_token.txt').read()

login(token=hf_token)

# Create the parser
parser = argparse.ArgumentParser(description='This is a simple argument parser example.')

# Add arguments
parser.add_argument('--no_ctx', action='store_true', help="Set this flag if you don't want to consider the context")

# Parse the arguments
parse_args = parser.parse_args()

wandb_key = open('./personal/wandb_token.txt').read()
wandb.login(key=wandb_key)
os.environ["WANDB_PROJECT"] = "fine-tuning-unveiling"


base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

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

data = process_dataset('llama')

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
        device_map="auto",
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

from peft import LoraConfig, TaskType

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
         output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
from peft import get_peft_model

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Assuming your model and tokenizer are already instantiated
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
tokenizer.pad_token = tokenizer.eos_token  # Make sure the pad token is set

# Instantiate the custom data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Continue setting up your TrainingArguments as before
training_args = TrainingArguments(
    output_dir='./ft_models/llama3',
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
    run_name="ft_llama3",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=data['val'],
    dataset_text_field="prompt",
    data_collator=data_collator,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length= 1024,
    packing= True,
)

trainer.train()
new_model = "llama3-ft"
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

if parse_args.no_ctx:
    test_lines = open('./data/llama_data/test_llama_no_ctx_lab.txt').readlines()
else:
    test_lines = open('./data/llama_data/test_llama_lab.txt').readlines()

# Clear the memory footprint
del model, trainer
torch.cuda.empty_cache()

# Reload the base model
base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map= {"": 0})

model = PeftModel.from_pretrained(base_model_reload, new_model)
model = model.merge_and_unload()

# Reload tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

results = []
from tqdm import tqdm
for prompt in tqdm(test_lines):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=1024, temperature=0.5, num_return_sequences=1)
    txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(txt)
    
    results.append([prompt, txt])

df_results = pd.DataFrame(results, columns=['prompt', 'results'])

path_res = "./ft_results/"
if parse_args.no_ctx:
    file_res = "{}_ft_results_no_ctx_lab.csv".format(base_model)
else:
    file_res = "{}_ft_results_lab.csv".format(base_model)

df_results.to_csv(os.path.join(path_res, file_res.split("/")[1]), index=False)