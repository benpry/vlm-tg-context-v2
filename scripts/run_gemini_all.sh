#!/bin/zsh
MODEL="gemini-3-flash-preview"

sbatch run_model_api.sh $MODEL --interactive --api_base https://generativelanguage.googleapis.com/ --n_samples 10 --n_trials 5