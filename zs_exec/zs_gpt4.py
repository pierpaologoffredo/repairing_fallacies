
from openai import OpenAI
import jsonlines
from tqdm import tqdm
import pandas as pd
import argparse
import os

# imports
import random
import time
import openai
# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

# Create the parser
parser = argparse.ArgumentParser(description='This is a simple argument parser example.')

# Add arguments
parser.add_argument('--no_ctx', action='store_true', help="Set this flag if you don't want to consider the context")

# Parse the arguments
parse_args = parser.parse_args()


openai_key = open('./personal/openai.txt').read()

client = OpenAI(
  organization='org-tTOkpZQV5A7Ls7taHeTW0Bnd',
  api_key=openai_key
)

model_name = "gpt-4"

if parse_args.no_ctx:
    data_path = "./data/gpt_data/zero_shot_gpt_no_ctx_lab.jsonl"
else:
    data_path = "./data/gpt_data/zero_shot_gpt_lab.jsonl"

# Opening JSON file
zs_lines = jsonlines.open(data_path)
results = []
for zs in tqdm(zs_lines):
    prompt = zs['messages']
    
    completion = completions_with_backoff(
        model = model_name,
        messages = prompt,
        temperature=0.2,
        max_tokens=512
    )
    output = completion.choices[0].message.content
    results.append([prompt[0]['content'], output])

df_results = pd.DataFrame(results, columns=['prompt', 'output'])

path_res = "./zs_results/"
if parse_args.no_ctx:
    file_res = "{}_zs_results_no_ctx_lab.csv".format(model_name)
else:
    file_res = "{}_zs_results_lab.csv".format(model_name)

df_results.to_csv(os.path.join(path_res, file_res), index=False)


