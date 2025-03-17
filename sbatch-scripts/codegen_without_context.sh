#!/bin/bash

#SBATCH --time=20:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1            # Number of nodes requested (1 nodes)
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --job-name="codegen_without_context"
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
  "Qwen/Qwen2.5-Coder-0.5B-Instruct"
  "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "Qwen/Qwen2.5-Coder-3B-Instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "meta-llama/CodeLlama-7b-Instruct-hf"
  "deepseek-ai/deepseek-coder-1.3b-instruct"
  "deepseek-ai/deepseek-coder-6.7b-instruct"
  "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
)

for model in "${models[@]}"; do
      echo "Starting evaluation for model: $model"

      python evals.py \
          --dataset_type code_generation \
          --data_file queries.jsonl \
          --model_names "$model" \
          --prompt_type standard \
          --eval_type codebertscore

      echo "Finished evaluation for model: $model"
      echo "----------------------------------------"
done