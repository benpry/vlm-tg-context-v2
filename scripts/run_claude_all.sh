#!/bin/zsh
MODEL="claude-sonnet-4-5"

BASE_ARGS="--api_base https://api.anthropic.com/v1"
ALL_ARGS=("" "--no_image" "--interactive" "--yoked")

for ARGS in "${ALL_ARGS[@]}"; do
    full_args="$BASE_ARGS $ARGS"
    sbatch run_model_api.sh $MODEL ${=full_args}
done