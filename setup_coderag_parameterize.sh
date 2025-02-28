#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=2:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --job-name="adis-install-coderag-deps"
#SBATCH --mail-user=azhar@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=instruction       # class node(s)
#SBATCH --account=s2025.coms.599.3    # account to use

# Parameterized variables
PROJECT_DIR=${PROJECT_DIR:-"/work/classtmp/azhar/projects/hpc-code-rag-bench"}
MICROMAMBA_ROOT_PATH=${MICROMAMBA_ROOT_PATH:-"/work/classtmp/azhar/micromamba"}
MICROMAMBA_ENV_NAME=${MICROMAMBA_ENV_NAME:-"coderag"}
MICROMAMBA_ENV_PATH="${MICROMAMBA_ROOT_PATH}/envs/${MICROMAMBA_ENV_NAME}"

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load micromamba
module load git
module load cuda

cd "$PROJECT_DIR"

echo "setting up crag"
echo "PROJECT_DIR is $PROJECT_DIR, MICROMAMBA_ROOT_PATH is $MICROMAMBA_ROOT_PATH, MICROMAMBA_ENV_NAME is $MICROMAMBA_ENV_NAME, MICROMAMBA_ENV_PATH is $MICROMAMBA_ENV_PATH"

eval "$(micromamba shell hook --shell=bash)"
micromamba env create -n "$MICROMAMBA_ENV_NAME" python=3.10.16 -c conda-forge -y -r "$MICROMAMBA_ROOT_PATH"
micromamba activate "$MICROMAMBA_ENV_PATH"

echo "coderag created"
# Check if 'coderag' environment works correctly
if micromamba activate "$MICROMAMBA_ENV_PATH"; then
    echo "Environment 'coderag' activated successfully."
    python -m pip install -r requirements.txt

    # List installed packages in the environment
    echo "Listing installed packages in 'coderag':"
    micromamba list
else
    echo "Failed to activate environment 'coderag'."
    exit 1
fi
