#!/bin/zsh
MODEL="claude-sonnet-4-5"

sbatch run_model_api.sh $MODEL --interactive --api_base https://api.anthropic.com/v1 --n_samples 10 --n_trials 5