#!/bin/bash
#SBATCH --time=4-0:00:00     # Maximum job runtime of 4 days
#SBATCH --cpus-per-task=1    # Number of processor cores per task
#SBATCH --nodes=1            # Number of nodes requested (1 node)
#SBATCH --partition=gpu      # Requesting the 'gpu' partition
#SBATCH --gres=gpu:1         # Requesting 1 GPU resource
#SBATCH --mem=512G           # Maximum memory per node (256 GB)
#SBATCH -J "Retrieval"     # Job name
#SBATCH --mail-user=arushi17@iastate.edu  # Email address for job notifications
#SBATCH --mail-type=BEGIN    # Send email at job start
#SBATCH --mail-type=END      # Send email at job end
#SBATCH --mail-type=FAIL     # Send email on job failure

module purge
module load zstd/1.5.5-haagkbq
module load micromamba
module load git

# Check if the first argument (PROJECTDIR) is provided
if [ -z "$1" ]; then
    echo "Error: No project directory supplied."
    exit 1  # Exit with an error code
else
    export PROJECTDIR=$1
    echo "Project directory is set to: $PROJECTDIR"
fi

# Set up micromamba root directory
export MAMBA_ROOT_PREFIX=$PROJECTDIR/micromamba

# Navigate to the project directory
cd $PROJECTDIR/hpc-code-rag-bench/retrieval

# Initialize micromamba and activate the environment
eval "$(micromamba shell hook --shell=bash)"
micromamba activate crag

# Check if 'crag' environment works correctly
if [ $? -eq 0 ]; then
    echo "Environment 'crag' activated successfully."

    # Step 1: Run the dataset preprocessing for HumanEval
    #python -m create.humaneval
    
    # Check if preprocessing was successful
    if [ $? -eq 0 ]; then
        echo "HumanEval dataset preprocessing completed successfully."

        # Step 2: Run the dense embedding models for retrieval
        python3 eval_beir_sbert_canonical.py \
		--model sentence-transformers/msmarco-MiniLM-L6-cos-v5 \
	        --dataset humaneval \
                --output_file $PROJECTDIR/results/humaneval_score.json \
                --results_file $PROJECTDIR/results/humaneval_retrieval.json

        if [ $? -eq 0 ]; then
            echo "Dense embedding model completed successfully."
        else
            echo "Error during dense embedding model retrieval."
            exit 1
        fi
    else
        echo "Error during HumanEval dataset preprocessing."
        exit 1
    fi
else
    echo "Failed to activate environment 'crag'."
    exit 1
fi

