import anthropic
import jsonlines
from tqdm import tqdm
import pandas as pd
import argparse
import os


# Create the parser
parser = argparse.ArgumentParser(description='This is a simple argument parser example.')

# Add arguments
parser.add_argument('--no_ctx', action='store_true', help="Set this flag if you don't want to consider the context")

# Parse the arguments
parse_args = parser.parse_args()

claude_key = open('./personal/claude.txt').read()

client = anthropic.Anthropic(
    api_key=claude_key,
)

if parse_args.no_ctx:
    data_path = "./data/gpt_data/zero_shot_gpt_no_ctx_lab.jsonl"
else:
    data_path = "./data/gpt_data/zero_shot_gpt_lab.jsonl"

model_name = "claude-3-opus-20240229"

# Opening JSON file
zs_lines = jsonlines.open(data_path)
results = []
for zs in tqdm(zs_lines):
    prompt = zs['messages']
    message = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=0.5,
        messages=prompt
    )

    output = message.content[0].text
    results.append([prompt[0]['content'], output])


df_results = pd.DataFrame(results, columns=['prompt', 'output'])

path_res = "./zs_results/"
if parse_args.no_ctx:
    file_res = "{}_zs_results_no_ctx_lab.csv".format(model_name)
else:
    file_res = "{}_zs_results_lab.csv".format(model_name)

df_results.to_csv(os.path.join(path_res, file_res), index=False)

