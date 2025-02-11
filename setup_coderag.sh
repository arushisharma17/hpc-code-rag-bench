#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=00:15:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --job-name="adis-install-coderag-deps"
#SBATCH --mail-user=azhar@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd /work/classtmp/azhar/projects/hpc-code-rag-bench

echo "setting up crag"

eval "$(micromamba shell hook --shell=bash)"
micromamba env create -n coderag python=3.10.16 -c conda-forge -y
micromamba activate coderag

echo "coderag created"
# Check if 'coderag' environment works correctly
if micromamba activate coderag; then
    echo "Environment 'coderag' activated successfully."
    python -m pip install -r requirements.txt

    # List installed packages in the environment
    echo "Listing installed packages in 'coderag':"
    micromamba list
else
    echo "Failed to activate environment 'coderag'."
    exit 1
fi

