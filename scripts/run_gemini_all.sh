ALL_ARGS=("--interactive" "" "--yoked")
model="gemini-3-flash-preview"
API_BASE="google"

for ARGS in "${ALL_ARGS[@]}"; do
    sbatch run_model_api.sh $model $ARGS --api_base $API_BASE
done