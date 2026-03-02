#!/bin/zsh
MODEL="gpt-5.2"

sbatch run_model_api.sh $MODEL --interactive --api_base https://api.openai.com/v1 --n_samples 10 --n_trials 5