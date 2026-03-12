#!/bin/zsh
MODEL="claude-sonnet-4-6"

sbatch run_model_api.sh $MODEL --interactive --api_base https://api.anthropic.com/v1 --n_samples 10
# sbatch run_model_api.sh $MODEL --api_base https://api.anthropic.com/v1 --n_samples 10
