import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

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



output_dir = "sql-code-llama/checkpoint-400"
base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

from peft import PeftModel
model = PeftModel.from_pretrained(model, output_dir)


eval_prompt = """You are an OpenMP High Performance Computing expert. Your job is to answer questions about performance optimization. You are given a question and code snippet regarding optimizating the performance.

You must output the answer to the given question.
### Input:
What is the performance issue in the given code snippet?

### Code Snippet:
#pragma omp parallel shared (a, b) private (c,d) \n { ... \n  #pragma omp critical \n { a += 2 * c; c = d * d; }}

### Response: 
Without the critical region, the first statement here leads to a data race. The second statement however involves private data only and unnecessarily increases the time taken to execute this construct
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
print("model.eval")

with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
