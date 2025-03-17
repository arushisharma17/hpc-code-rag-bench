#!/bin/bash

# e.g. to run:
# ./run_codegen.sh --partition="instruction" --account="s2025.coms.599.3" --gres="gpu:a100:1" --mail-user="azhar@iastate.edu" --project-dir="/work/classtmp/azhar/projects/hpc-code-rag-bench" --micromamba-root-path="/work/classtmp/azhar/micromamba" --micromamba-env-name="hpccoderag" --hf-home-path="/work/classtmp/azhar/hf"

# default param values
partition="instruction"
account="s2025.coms.599.3"
gres="gpu:a100:1"
mail_user="azhar@iastate.edu"
project_dir="/work/classtmp/azhar/projects/hpc-code-rag-bench"
micromamba_root_path="/work/classtmp/azhar/micromamba"
micromamba_env_name="hpccoderag"
hf_home_path="/work/classtmp/azhar/hf"

# parse params from cli
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --partition=*) partition="${1#*=}"; shift ;;
        --account=*) account="${1#*=}"; shift ;;
        --gres=*) gres="${1#*=}"; shift ;;
        --mail-user=*) mail_user="${1#*=}"; shift ;;
        --project-dir=*) project_dir="${1#*=}"; shift ;;
        --micromamba-root-path=*) micromamba_root_path="${1#*=}"; shift ;;
        --micromamba-env-name=*) micromamba_env_name="${1#*=}"; shift ;;
        --hf-home-path=*) hf_home_path="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# codegen sripts we wanna run
sbatch_scripts=(
    "codegen_github_codellama.sh"
    "codegen_github_llama.sh"
    "codegen_stackoverflow_codellama.sh"
    "codegen_stackoverflow_llama.sh"
    "codegen_without_context.sh"
    "codegen_github_deepseekcoder.sh"
    "codegen_github_qwencoder.sh"
    "codegen_stackoverflow_deepseekcoder.sh"
    "codegen_stackoverflow_qwencoder.sh"
)

# submit to slurm
for script in "${sbatch_scripts[@]}"; do
    echo "Submitting job: $script with partition=$partition, account=$account, gres=$gres, mail-user=$mail_user, project-dir=$project_dir, micromamba-root-path=$micromamba_root_path, micromamba-env-name=$micromamba_env_name, hf-home-path=$hf_home_path"
    sbatch --export=PROJECT_DIR="$project_dir",MICROMAMBA_ROOT_PATH="$micromamba_root_path",MICROMAMBA_ENV_NAME="$micromamba_env_name",HF_HOME_PATH="$hf_home_path" --partition="$partition" --account="$account" --gres="$gres" --mail-user="$mail_user" "$script"
done
