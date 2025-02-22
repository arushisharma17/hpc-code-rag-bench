#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=5:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=4            # Number of nodes requested (4 nodes)
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --job-name="adis-generate-without-retrieval"
#SBATCH --mail-user=azhar@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=instruction       # class node(s)
#SBATCH --account=s2025.coms.599.3    # account to use
#SBATCH --gres=gpu:a100:1   # Required GPU hardware


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load micromamba
module load git
module load cuda

cd /work/classtmp/azhar/projects/hpc-code-rag-bench/generation

micromamba activate coderag

echo "running generation..."

# generation without retrieval
HF_HOME=/work/classtmp/azhar/hf OPENAI_API_KEY=dummy-for-now API_KEY=dummy-for-now python main.py --tasks "humaneval" \
--model "bigcode/starcoder2-7b" \
--dataset_path "openai_humaneval" \
--allow_code_execution


echo "generation finished..."