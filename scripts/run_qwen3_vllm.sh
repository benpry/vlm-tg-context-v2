#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops,sc-loprio
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-output/qwen3_%j.out
#SBATCH --error=slurm-output/qwen3_%j.err
#SBATCH --constraint=[80G|141G]

source ~/.zshrc
cd ~/vlm-tg-context

# install conda if needed, create vtc environment if needed, activate vtc
source scripts/set_up_uv.sh

MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"

vllm serve --model $MODEL_NAME --host 0.0.0.0 --port 8000 &

sleep 5

python scripts/call_lm.py \
    --model_name $MODEL_NAME \
    --data_dir full_feedback \
    --api_base localhost:8000
