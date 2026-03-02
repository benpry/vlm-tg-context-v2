MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"

ALL_ARGS=("" "--interactive" "--yoked" "--no_image")

for ARGS in "${ALL_ARGS[@]}"; do
    sbatch run_model.sh $MODEL $ARGS
done