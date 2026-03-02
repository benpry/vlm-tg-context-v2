MODELS=("Qwen/Qwen2.5-VL-32B-Instruct" "google/gemma-3-27b-it" "moonshotai/Kimi-VL-A3B-Instruct" "Qwen/Qwen3-VL-32B-Instruct" "allenai/Molmo2-8B")

ARGS="--yoked"

for model in ${MODELS[@]}; do
    sbatch run_model.sh $model $ARGS
done
