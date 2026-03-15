#!/bin/zsh
# Rerun all frontier models on banana-affected rows

# sbatch scripts/rerun_banana_api.sh claude-sonnet-4-6 --api_base https://api.anthropic.com/v1 --n_samples 10
# sbatch scripts/rerun_banana_api.sh gpt-5.2 --api_base https://api.openai.com/v1 --n_samples 10
sbatch scripts/rerun_banana_api.sh gemini-3-flash-preview --api_base https://generativelanguage.googleapis.com/ --n_samples 10
