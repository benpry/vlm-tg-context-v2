#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --exclude=cocoflops-hgx-1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-output/banana_rerun_%j.out
#SBATCH --error=slurm-output/banana_rerun_%j.err

source ~/.zshrc
cd ~/vlm-tg-context

# create the environment if needed and activate it
source scripts/set_up_uv.sh

MODEL_NAME=$1
shift
EXTRA_ARGS=("$@")

echo "model name: $MODEL_NAME"
echo "extra args: ${EXTRA_ARGS}"

python scripts/rerun_banana_rows.py \
    --model_name $MODEL_NAME \
    $EXTRA_ARGS
