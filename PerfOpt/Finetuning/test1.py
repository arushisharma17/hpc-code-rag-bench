from datetime import datetime
import os
import sys

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq


from datasets import load_dataset


dataset = load_dataset("b-mc2/sql-create-context", split="train")
train_dataset = dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]
print(train_dataset[3])

base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")



eval_prompt = """You are an OpenMP High Performance Computing expert. Your job is to answer questions about performance optimization. You are given a question and code snippet regarding optimizating the performance.

You must output the answer to the given question.
### Input:
What is the performance issue in the given code snippet?

### Code Snippet:
#pragma omp parallel shared (a, b) private (c,d) \n { ... \n  #pragma omp critical \n { a += 2 * c; c = d * d; }}

### Response:
"""

### Response: 
#Without the critical region, the first statement here leads to a data race. The second statement however involves private data only and unnecessarily increases the time taken to execute this construct

'''
eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""
'''

# {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=600)[0], skip_special_tokens=True))

