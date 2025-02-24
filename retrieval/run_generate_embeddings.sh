#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=12:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1            # Number of nodes requested (1 nodes)
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --job-name="gen-embeddings"
#SBATCH --mail-user=azhar@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=instruction       # class node(s)
#SBATCH --account=s2025.coms.599.3    # account to use
#SBATCH --gres=gpu:a100:2   # Required GPU hardware


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load micromamba
module load git
module load cuda

cd /work/classtmp/azhar/projects/hpc-code-rag-bench/retrieval

micromamba activate coderag

# Define models and datasets as arrays
models=("BAAI/bge-m3" "jinaai/jina-embeddings-v2-base-code" "flax-sentence-embeddings/st-codesearch-distilroberta-base", "thenlper/gte-large", "Salesforce/SFR-Embedding-Code-2B_R", "sentence-transformers/all-MiniLM-L6-v2")
datasets=("code-rag-bench/programming-solutions" "code-rag-bench/online-tutorials", "code-rag-bench/library-documentation", "code-rag-bench/stackoverflow-posts", "code-rag-bench/github-repos-python", "code-rag-bench/github-repos")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    output_dir="embeddings-$(basename "$model")-$(basename "$dataset")"

    echo "Running: Model=$model, Dataset=$dataset, Output=$output_dir"

    python generate_embeddings.py \
      --model "$model" \
      --output_dir "$output_dir" \
      --prefix programming-solutions \
      --hf_datasets "$dataset" \
      --shard_id 0 \
      --num_shards 1
  done
done