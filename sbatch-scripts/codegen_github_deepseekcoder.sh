#!/bin/bash

#SBATCH --time=10:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1            # Number of nodes requested (1 nodes)
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --job-name="codegen_github_deepseekcoder"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:a100:1   # Required GPU hardware


PROJECT_DIR=${PROJECT_DIR:-"/work/classtmp/azhar/projects/hpc-code-rag-bench"}
MICROMAMBA_ROOT_PATH=${MICROMAMBA_ROOT_PATH:-"/work/classtmp/azhar/micromamba"}
MICROMAMBA_ENV_NAME=${MICROMAMBA_ENV_NAME:-"hpccoderag"}
MICROMAMBA_ENV_PATH="${MICROMAMBA_ROOT_PATH}/envs/${MICROMAMBA_ENV_NAME}"
HF_HOME_PATH=${HF_HOME_PATH:-"/work/classtmp/azhar/hf"}

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load micromamba
module load git
module load cuda

echo "PROJECT_DIR is $PROJECT_DIR, MICROMAMBA_ROOT_PATH is $MICROMAMBA_ROOT_PATH, MICROMAMBA_ENV_NAME is $MICROMAMBA_ENV_NAME, MICROMAMBA_ENV_PATH is $MICROMAMBA_ENV_PATH, HF_HOME_PATH is $HF_HOME_PATH"
cd "$PROJECT_DIR/PerfOpt/Evaluation" || exit
echo "current workdir is $(pwd)"

eval "$(micromamba shell hook --shell=bash)"
micromamba activate "$MICROMAMBA_ENV_PATH"

models=(
  "deepseek-ai/deepseek-coder-1.3b-instruct"
  "deepseek-ai/deepseek-coder-6.7b-instruct"
  "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
)
retrieval_paths=(
  "datasets/github/GIST-Embedding-v0.json"
  "datasets/github/st-codesearch-distilroberta-base.json"
)
corpus_path="datasets/github/corpus.jsonl"

for model in "${models[@]}"; do
    for retrieval in "${retrieval_paths[@]}"; do
        echo "Starting evaluation for model: $model with retrieval path: $retrieval"

        HF_HOME=$HF_HOME_PATH python evals.py \
            --dataset_type code_generation \
            --data_file queries.jsonl \
            --model_names "$model" \
            --prompt_type standard \
            --eval_type codebertscore \
            --rag \
            --k_documents 3 \
            --retrieval_path "$retrieval" \
            --corpus_path "$corpus_path"

        echo "Finished evaluation for model: $model with retrieval path: $retrieval"
        echo "----------------------------------------"
    done
done
