#!/bin/zsh
# Rerun Llama 3.2 with untruncated logprobs (REPORT_banana_verification.md §6.2).
#
# The archived Llama results were produced by vLLM 0.10.1.1's V0 engine, which
# reports logprobs after the top_p = 0.9 nucleus from the model's
# generation_config.json had been applied. src/lm.py now pins top_p = 1 and
# top_k = -1 in every logprob request, and src/utils.py refuses masked
# logprobs, so this rerun produces the model's full distribution.
#
# Results go to OUTPUT_ROOT, mirroring data/logprobs/; the archive under
# data/logprobs is not touched. The four jobs are chained (afterany) because two
# jobs syncing the same uv environment on one node race each other.
#
# Usage: zsh scripts/rerun_llama.sh

set -e
MODEL_NAME=meta-llama/Llama-3.2-11B-Vision-Instruct
OUTPUT_ROOT=/juice2/scr2/benpry/vlm-tg-context-logprobs-llama-rerun

# Submit from scripts/ so the job logs land in scripts/slurm-output/ like the
# original runs (run_model.sh cd's to the project root itself).
cd ~/vlm-tg-context/scripts
mkdir -p $OUTPUT_ROOT slurm-output

previous_job=""
for mode_args in "" "--no_image" "--yoked" "--interactive"; do
    dependency=()
    if [[ -n $previous_job ]]; then
        dependency=(--dependency=afterany:$previous_job)
    fi
    # $mode_args is unquoted on purpose: an empty value must vanish, not become "".
    job_id=$(sbatch --parsable $dependency run_model.sh $MODEL_NAME ${=mode_args} --output_root $OUTPUT_ROOT)
    echo "submitted job $job_id: run_model.sh $MODEL_NAME ${mode_args:-(full feedback, image)} --output_root $OUTPUT_ROOT"
    previous_job=$job_id
done
echo "When all four have finished, validate with:"
echo "  python scripts/check_llama_rerun.py"
