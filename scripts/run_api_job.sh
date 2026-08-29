#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-output/api_job_%j.out
#SBATCH --error=slurm-output/api_job_%j.err

# Run one API-only (frontier model) command in the existing vtc environment on
# cocoflops1, without re-syncing it: set_up_uv.sh's `uv sync` races with any
# other job syncing at the same time, and API runs need no GPU or vLLM.
#
# Usage (from scripts/, so the logs land in scripts/slurm-output/):
#   sbatch run_api_job.sh python scripts/call_lm.py --model_name gemini-3-flash-preview ...
#   sbatch run_api_job.sh python scripts/rerun_sessions.py --model_name claude-sonnet-4-6 ...

source ~/.zshrc
cd ~/vlm-tg-context

# LANGCOG_GEMINI_API_KEY is dead; src/clients.py would try it first.
unset LANGCOG_GEMINI_API_KEY

source /scr/benpry/uv/vtc/bin/activate
# Any job that sources set_up_uv_llama.sh runs `uv sync` against this (project)
# environment and strips the editable install of src; do not depend on it.
export PYTHONPATH=~/vlm-tg-context
python -c "import src.lm, anthropic, google.genai" || { echo "ERROR: vtc environment unusable" >&2; exit 1; }
for var in COCOLAB_GEMINI_API_KEY ANTHROPIC_API_KEY; do
    [[ -n ${(P)var} ]] || echo "WARNING: $var is not set"
done

echo "running: $@"
"$@"
