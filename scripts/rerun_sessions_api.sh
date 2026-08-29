#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --exclude=cocoflops-hgx-1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-output/session_rerun_%j.out
#SBATCH --error=slurm-output/session_rerun_%j.err

# Rerun sessions of a frontier model's interactive-yoked evaluation (see
# scripts/rerun_sessions.py). Mirrors rerun_banana_api.sh.
# Usage: sbatch scripts/rerun_sessions_api.sh <model_name> --api_base <url> --sessions_file <json> [--n_samples 10]

source ~/.zshrc
cd ~/vlm-tg-context

# LANGCOG_GEMINI_API_KEY is dead; src/clients.py tries it first.
unset LANGCOG_GEMINI_API_KEY

# create the environment if needed and activate it
source scripts/set_up_uv.sh

MODEL_NAME=$1
shift
EXTRA_ARGS=("$@")

echo "model name: $MODEL_NAME"
echo "extra args: ${EXTRA_ARGS}"

python scripts/rerun_sessions.py \
    --model_name $MODEL_NAME \
    $EXTRA_ARGS
