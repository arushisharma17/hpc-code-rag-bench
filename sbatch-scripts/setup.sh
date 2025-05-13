#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

# e.g. to run:
# sbatch --export=PROJECT_DIR="/work/classtmp/azhar/projects/hpc-code-rag-bench",MICROMAMBA_ROOT_PATH="/work/classtmp/azhar/micromamba",MICROMAMBA_ENV_NAME="hpccoderag",HF_HOME_PATH="/work/classtmp/azhar/hf" \
  #--partition="instruction" \
  #--account="s2025.coms.599.3" \
  #--mail-user="azhar@iastate.edu" setup.sh

#SBATCH --time=2:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --job-name="setup"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Parameterized variables
PROJECT_DIR=${PROJECT_DIR:-"/work/classtmp/azhar/projects/hpc-code-rag-bench"}
MICROMAMBA_ROOT_PATH=${MICROMAMBA_ROOT_PATH:-"/work/classtmp/azhar/micromamba"}
MICROMAMBA_ENV_NAME=${MICROMAMBA_ENV_NAME:-"hpccoderag"}
MICROMAMBA_ENV_PATH="${MICROMAMBA_ROOT_PATH}/envs/${MICROMAMBA_ENV_NAME}"

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load micromamba
module load git
module load cuda

echo "setting up $MICROMAMBA_ENV_NAME"
echo "PROJECT_DIR is $PROJECT_DIR, MICROMAMBA_ROOT_PATH is $MICROMAMBA_ROOT_PATH, MICROMAMBA_ENV_NAME is $MICROMAMBA_ENV_NAME, MICROMAMBA_ENV_PATH is $MICROMAMBA_ENV_PATH"

cd "$PROJECT_DIR/PerfOpt" || exit
echo "current workdir is $(pwd)"


eval "$(micromamba shell hook --shell=bash)"
micromamba env create -n "$MICROMAMBA_ENV_NAME" python=3.10.16 -c conda-forge -y -r "$MICROMAMBA_ROOT_PATH"
micromamba activate "$MICROMAMBA_ENV_PATH"

echo "$MICROMAMBA_ENV_NAME created"

if micromamba activate "$MICROMAMBA_ENV_PATH"; then
    echo "Environment $MICROMAMBA_ENV_NAME activated successfully."
    python -m pip install -r requirements.txt

    # List installed packages in the environment
    echo "Listing installed packages in $MICROMAMBA_ENV_NAME:"
    micromamba list
else
    echo "Failed to activate environment $MICROMAMBA_ENV_NAME."
    exit 1
fi
