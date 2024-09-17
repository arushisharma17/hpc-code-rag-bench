#!/bin/bash
#SBATCH --time=4-0:00:00     # Maximum job runtime of 4 days
#SBATCH --cpus-per-task=1    # Number of processor cores per task
#SBATCH --nodes=4            # Number of nodes requested (4 nodes)
#SBATCH -J "Setup"           # Job name
#SBATCH --mail-user=arushi17@iastate.edu  # Email address for job notifications
#SBATCH --mail-type=BEGIN    # Send email at job start
#SBATCH --mail-type=END      # Send email at job end
#SBATCH --mail-type=FAIL     # Send email on job failure

module purge
module load micromamba
module load git

# Check if the first argument is provided for PROJECTDIR
if [ -z "$1" ]; then
    echo "Error: No project directory supplied."
    echo "Usage: ./setup.sh /path/to/LatentConceptAnalysis"
    exit 1  # Exit with an error code
else
    export PROJECTDIR=$1
    echo "Project directory is set to: $PROJECTDIR"
fi

cd $PROJECTDIR

# Setup micromamba
if [ ! -d "micromamba" ]; then
    mkdir micromamba
fi
export MAMBA_ROOT_PREFIX=$PROJECTDIR/micromamba

cd hpc-code-rag-bench

echo "setting up crag" 

eval "$(micromamba shell hook --shell=bash)"
micromamba env create -n crag python=3.10 -c conda-forge -y
micromamba activate crag

echo "crag created"
# Check if 'crag' environment works correctly
if micromamba activate crag; then
    echo "Environment 'crag' activated successfully."
    python -m pip install -r requirements.txt

    # List installed packages in the environment
    echo "Listing installed packages in 'crag':"
    micromamba list
else
    echo "Failed to activate environment 'crag'."
    exit 1
fi


