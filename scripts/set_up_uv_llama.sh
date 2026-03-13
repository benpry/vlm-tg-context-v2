if [ ! -d "${SCR_ROOT_DIR}/uv/vtc-llama" ]; then
    mkdir -p "${SCR_ROOT_DIR}/uv"
    uv venv ${SCR_ROOT_DIR}/uv/vtc-llama --clear
fi

# activate the environment
source ${SCR_ROOT_DIR}/uv/vtc-llama/bin/activate
uv sync
uv pip install -e .
uv pip install vllm==0.10.1.1 transformers==4.56.2 torch==2.7.1