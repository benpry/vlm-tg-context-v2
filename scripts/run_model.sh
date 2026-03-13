#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-output/run_model_%j.out
#SBATCH --error=slurm-output/run_model_%j.err
#SBATCH --constraint=[40G|48G|80G|141G]

source ~/.zshrc
cd ~/vlm-tg-context

MODEL_NAME=$1
shift
EXTRA_ARGS=("$@")

echo "model name: $MODEL_NAME"
echo "extra args: ${EXTRA_ARGS}"

if [[ $MODEL_NAME == "meta-llama/Llama-3.2-11B-Vision-Instruct" ]]; then
    source scripts/set_up_uv_llama.sh
else

    source scripts/set_up_uv.sh
fi

# Pick a unique port per job (override with PORT env var if set)
PORT=${PORT:-$((8000 + (${SLURM_JOB_ID:-0} % 1000)))}
API_BASE="http://localhost:${PORT}/v1"
echo "using port: $PORT (api_base: $API_BASE)"

vllm serve $MODEL_NAME \
 --tensor-parallel-size 2 \
 --dtype bfloat16 \
 --host 0.0.0.0 \
 --port ${PORT} \
 --limit-mm-per-prompt '{"image":1}' \
 --max-model-len 12288 \
 --max-num-batched-tokens 12288 \
 --gpu-memory-utilization 0.95 \
 --max-num-seqs 8 \
 --max-logprobs 1000 \
 --trust-remote-code > /dev/null &

sleep 5m

python scripts/call_lm.py \
    --model_name $MODEL_NAME \
    --api_base $API_BASE \
    $EXTRA_ARGS 
