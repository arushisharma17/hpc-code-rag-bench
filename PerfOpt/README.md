# PerfOpt-Dataset-and-OptiAdvisor-Tool

## Description
This repository contains the code for the paper "Are Large Language Models Effective to Guide Analysis and Optimizations for Parallel Computing?" It includes tools for dataset generation, scaling, evaluation, and fine-tuning.

### PerfOpt Dataset Statistics

| Dataset Type | Dataset File | Number of Entries |
|--------------|--------------|-------------------|
| [Open-Ended](https://huggingface.co/datasets/sharmaarushi17/HPCPerfOpt-Open-ended/tree/main) | `Open ended text.csv` | 29 |
| [Open-Ended](https://huggingface.co/datasets/sharmaarushi17/HPCPerfOpt-Open-ended/tree/main) | `Rodinia-open-ended-basic.csv` | 50 |
| [Open-Ended](https://huggingface.co/datasets/sharmaarushi17/HPCPerfOpt-Open-ended/tree/main) | `Rodinia-open-ended-advanced.csv` | 30 |
| [Open-Ended](https://huggingface.co/datasets/sharmaarushi17/HPCPerfOpt-Open-ended/tree/main) | `code.csv` | 63 |
| [MCQ](https://huggingface.co/datasets/sharmaarushi17/HPCPerfOpt-MCQA/tree/main) | `mcq-single.csv` | 88 |
| [MCQ](https://huggingface.co/datasets/sharmaarushi17/HPCPerfOpt-MCQA/tree/main) | `mcq-rodinia-basic.csv` | 40 |
| [MCQ](https://huggingface.co/datasets/sharmaarushi17/HPCPerfOpt-MCQA/tree/main) | `mcq-rodinia-advanced.csv` | 40 |




## Features
- **Dataset Generation and Scaling:** Tools to create and scale datasets for model training.
- **Evaluation:** Scripts to evaluate model performance on various datasets.
- **Fine-tuning:** Resources to fine-tune models for specific tasks.

## Prerequisites
Before getting started, make sure to set up the following prerequisite:
- [LM4HPC](https://github.com/HPC-FAIR/LM4HPC/tree/codellama): Library for managing and utilizing large language models in HPC environments.

### Create virtual environment and install LM4HPC on Perlmutter

```
export HF_HOME=<path to huggingface cache>
module load pytorch/2.0.1 
export PYTHONUSERBASE="<path to loaded pytorch container environment>"
cd $SCRATCH
python -m venv <name of virtual environment for eg. lm4hpcenv>
cd lm4hpcenv
source lm4hpcenv/bin/activate
pip install -q transformers torch torchvision accelerate openai PyPDF2 deeplake tiktoken langchain sentence_transformers InstructorEmbedding nltk code_bert_score
pip install -i https://test.pypi.org/simple/ bitsandbytes

pip uninstall openai
pip install openai==0.28

python -m pip -q install -e .
```

## Implementation Steps on Perlmutter

### Sample Script for Perlmutter
Use this script as a template to run your tasks on the Perlmutter cluster:

```bash
#!/bin/bash
#SBATCH -A m2956_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 15:00:00
#SBATCH -N 1
#SBATCH -c 32

export HF_HOME=<path to huggingface cache>
module load pytorch/2.0.1 
export PYTHONUSERBASE="<path to loaded pytorch container environment>"

cd $SCRATCH
export OPENAI_API_KEY='YOUR KEY HERE'
export WANDB_API_KEY='YOUR KEY HERE'

source ~/.bashrc
source hpcenv/bin/activate
wandb login
huggingface-cli whoami
```

### Directory Structure

The project is organized into several directories, as follows:

- **`Root`**: The root directory of the project.
  - **`Dataset_Scaling`**: This directory contains scripts and tools for scaling datasets.
    - `dataset_scaling.py`: Python script for scaling the dataset.
    - **`Outputs`**: Directory where the scaled datasets will be saved.
    - `scaled_data`: [Description or contents of 'scaled_data']
  - **`Evaluation`**: This directory contains scripts and tools for evaluating datasets.
    - **`Results`**: Root directory for all evaluation results.
      - **`[Dataset Type]`**: A subdirectory for each type of dataset being evaluated (e.g., "open-ended", "MCQ").
        - **`[Data File]`**: Under each dataset type, a subdirectory for each data file used in the evaluations.
          - **`[Model Name]`**: Inside each data file directory, further subdirectories named after the models evaluated.
            - `CSV Files`: The results are stored in CSV files within each model's directory.
  - **`Finetuning`**: This directory contains scripts and tools for finetuning the codellama model on the open-ended dataset.
    - **`sql-codellama`**: All the checkpoints of the finetuned model are stored here.



## Dataset Scaling

- To scale the dataset, follow these steps:
  - Navigate to the Dataset Scaling directory:
    ```bash
    cd Dataset_Scaling
    ```
    
  - Run the scaling script with the desired model. For example, for GPT-4:
    ```bash
    python dataset_scaling.py --model_names gpt-4
    ```
### TODO's and improvements
- [ ] Use GPT-4-Turbo JSON mode to streamling output of generated questions


## Retrieval

- Before using retrieval, a datastore must be created with the following steps:
  - Run
    ```bash
    python3 -m create.{"py_file"}
    ```
  - where py_file is one of the create functions in hpc-code-rag-bench
  - This creates a /dataset directory with corpus.jsonl, query.jsonl and qrels. For the purposes of using retrieval, we only care about query and corpus where corpus is our list of context information for the datastore.
  - Note that the query will be generated with the provided datasets only and for an isolated query file, we will need to parse the data separately
### TODO
- Create function to create corpus and query files separately to facilitate testing on data that is not contained in the datastore.
- Create function to merge corpus libraries from multiple datasets and/or multiple runs of the create function

## Evaluation

- To run the evaluation code on different datasets and models, follow these steps:
  - Navigate to the Evaluation directory:
    ```bash
    cd Evaluation
    ```
  - evals.py: This script evaluates models using datasets from the Hugging Face Hub. It supports various types of datasets and models, and allows for different modes of operation and evaluation criteria. To run the script with a specific dataset, model, and evaluation type, use a command like:
    ```bash
    python evals.py --dataset_type mcq --data_file mcq-single.csv --model_names gpt-4 --test_mode --prompt_type standard --eval_type exact_match
    ```
  - Usage: To use this script, you will need to specify several command line arguments that control its behavior. Below are the details of these arguments:

    1. `--test_mode`: Run in test mode for a smaller dataset sample. Usage: `--test_mode`
    
    2. `--dataset_type` (Required): Choose the dataset type: `mcq` for multiple choice questions or `open_ended` for open-ended questions. Usage: `--dataset_type mcq` or `--dataset_type open_ended`
    
    3. `--data_file` (Required): Specify the dataset file from the Hugging Face Hub. For `mcq`, use files like "mcq-single.csv", "rodinia-basic.csv", etc., and for `open_ended`, use "text.csv" or "code.csv". Usage: `--data_file filename.csv`
    
    4. `--model_names` (Required): Name(s) of the models for evaluation. Provide multiple names separated by spaces. Usage: `--model_names model1 model2`
    
    5. `--prompt_type`: Choose the prompt type for the LLM (default is 'none'). Options include `standard`, `cot`, `text`, `code`, or `none`. Usage: `--prompt_type text`
    
    6. `--eval_type`: Select the evaluation type (default is 'none'). Options are `exact_match`, `semantic_similarity`, `bleu_score`, `codebertscore`, `LLM_as_a_judge`. Usage: `--eval_type semantic_similarity`
	
	7. `--rag`: Enable model prompting with retrieval results. If passed, two files are required, /results/corpus.jsonl and /results/retrieval.json


### TODO's and improvements

- [ ] Restrict output for evaluation with this pipeline 


  ## Finetuning CodeLLAMA

  
