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

from datasets import load_dataset,concatenate_datasets


def load_dataset_from_hub(dataset_type, data_file, test_mode):
    """
    Load a dataset from the Hugging Face Hub based on the dataset type and file provided.

    Parameters:
    - dataset_type (str): The type of dataset to load. Accepted values are "mcq" for multiple-choice questions
                          and "open_ended" for open-ended questions.
    - data_file (str): The filename or path of the dataset file to load. This file should be located in the
                       Hugging Face Hub repository specified by the dataset name.

                       For "mcq" type, expected files could be 'mcq-single.csv', 'rodinia-basic.csv', or
                       'rodinia-advanced.csv'.

                       For "open_ended" type, expected files could be 'text.csv' or 'code.csv'.
    - test_mode (bool): If True, only load the first two examples from the dataset for testing purposes.

    Returns:
    - dataset (DatasetDict): A `datasets.DatasetDict` object containing the loaded dataset.

    Raises:
    - ValueError: If the dataset_type is not one of the accepted values.

    Example usage:
    mcq_dataset = load_dataset_from_hub("mcq", "mcq-single.csv")
    open_ended_dataset = load_dataset_from_hub("open_ended", "text.csv")
    """

    # Load MCQ-type dataset from the HPCPerfOpt-MCQA repository on the Hugging Face Hub.
    if dataset_type == "mcq":
        dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files=data_file, split = "train")

    # Load open-ended dataset from the HPCPerfOpt-Open-ended repository on the Hugging Face Hub.
    elif dataset_type == "open_ended":
        dataset = load_dataset("sharmaarushi17/HPCPerfOpt-Open-ended", data_files=data_file, split = "train")
        print (type(dataset))
    # Raise an error if an invalid dataset_type is provided.
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

   # If test_mode is True, only take the first two examples from each split.
   # if test_mode:
        #for split in dataset.keys():
        #    dataset[split] = dataset[split].select(range(2))

    return dataset


train_dataset1 = load_dataset_from_hub("open_ended", "rodinia-open-ended-basic.csv", True)
train_dataset2 = load_dataset_from_hub("open_ended", "rodinia-open-ended-advanced.csv", True)
print(type(train_dataset1))
print(type(train_dataset2))
train_dataset = concatenate_datasets([train_dataset1,train_dataset2])

eval_dataset = load_dataset_from_hub("open_ended", "text.csv", True)
print(f"Loaded dataset with {len(train_dataset)} examples for training.",train_dataset)
print(f"Loaded dataset with {len(eval_dataset)} examples for evaluation.",train_dataset)


#dataset = load_dataset("b-mc2/sql-create-context", split="train")
#train_dataset = dataset.train_test_split(test_size=0.1)["train"]
#eval_dataset = dataset.train_test_split(test_size=0.1)["test"]
#print(train_dataset[3])
                                                             

base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")


#For baseline evaluation 
eval_prompt = """You are an OpenMP High Performance Computing expert. Your job is to answer questions about performance optimization. You are given a question and code snippet regarding optimizating the performance.

You must output the answer to the given question.
### Input:
What is the performance issue in the given code snippet?

### Code Snippet:
#pragma omp parallel shared (a, b) private (c,d) \n { ... \n  #pragma omp critical \n { a += 2 * c; c = d * d; }}

### Response: 
Without the critical region, the first statement here leads to a data race. The second statement however involves private data only and unnecessarily increases the time taken to execute this construct
"""
#_________________
"""
#eval_prompt = You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

#You must output the SQL query that answers the question.
### Input:
#Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

### Context:
#CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
# {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}
"""
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))


tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""You are an OpenMP High Performance Computing expert. Your job is to answer questions about performance optimization. You are given a question and code snippet regarding optimizating the performance.

You must output the answer to the given question.

### Input:
{data_point["Question"]}

### Context:
{data_point["Code Snippet"]}

### Response:
{data_point["Answer"]}
"""
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)


model.train() # put model back into training mode
model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

resume_from_checkpoint = "" # set this to the adapter_model.bin file you want to resume from

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

wandb_project = "sql-try2-coder"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

batch_size = 128
per_device_train_batch_size = 8
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "sql-code-llama"

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=400,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)


trainer.train()

