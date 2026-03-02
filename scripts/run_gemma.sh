#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops,sc-loprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=slurm-output/gemma_%j.out
#SBATCH --error=slurm-output/gemma_%j.err
#SBATCH --constraint=[80G|141G]

source ~/.zshrc
cd ~/vlm-tg-context

# install conda if needed, create vtc environment if needed, activate vtc
source scripts/set_up_uv.sh

MODEL_NAME="google/gemma-3-27b-it"

vllm serve $MODEL_NAME --host 0.0.0.0 --port 8000 --tensor-parallel-size 2

python scripts/call_lm.py \
    --model_name $MODEL_NAME \
    --api_base http://localhost:8000 \
    --data_dir full_feedback \
    --interactive \
    --overwrite
