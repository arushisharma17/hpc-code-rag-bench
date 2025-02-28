#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=5:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1            # Number of nodes requested (1 nodes)
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --job-name="adis-generate-without-retrieval"
#SBATCH --mail-user=azhar@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=instruction       # class node(s)
#SBATCH --account=s2025.coms.599.3    # account to use
#SBATCH --gres=gpu:a100:1   # Required GPU hardware


PROJECT_DIR=${PROJECT_DIR:-"/work/classtmp/azhar/projects/hpc-code-rag-bench"}
MICROMAMBA_ROOT_PATH=${MICROMAMBA_ROOT_PATH:-"/work/classtmp/azhar/micromamba"}
MICROMAMBA_ENV_NAME=${MICROMAMBA_ENV_NAME:-"coderag"}
MICROMAMBA_ENV_PATH="${MICROMAMBA_ROOT_PATH}/envs/${MICROMAMBA_ENV_NAME}"
HF_HOME_PATH=${HF_HOME_PATH:-"/work/classtmp/azhar/hf"}
MODEL=${MODEL:-"bigcode/starcoder2-7b"}
DATASET_PATH=${DATASET_PATH:-"openai_humaneval"}
TASKS=${TASKS:-"humaneval"}

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load micromamba
module load git
module load cuda

cd "$PROJECT_DIR/generation"

echo "PROJECT_DIR is $PROJECT_DIR, MICROMAMBA_ROOT_PATH is $MICROMAMBA_ROOT_PATH, MICROMAMBA_ENV_NAME is $MICROMAMBA_ENV_NAME, MICROMAMBA_ENV_PATH is $MICROMAMBA_ENV_PATH, HF_HOME_PATH is $HF_HOME_PATH, MODEL is $MODEL, DATASET_PATH is $DATASET_PATH, TASKS is $TASKS"

eval "$(micromamba shell hook --shell=bash)"
micromamba activate "$MICROMAMBA_ENV_PATH"

echo "running generation..."

# generation without retrieval
HF_HOME="$HF_HOME_PATH" OPENAI_API_KEY=dummy-for-now API_KEY=dummy-for-now python main.py \
--tasks "$TASKS" \
--model "$MODEL" \
--dataset_path "$DATASET_PATH" \
--allow_code_execution


echo "generation finished..."