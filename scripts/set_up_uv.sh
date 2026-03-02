if [ ! -d "${SCR_ROOT_DIR}/uv/vtc" ]; then
    mkdir -p "${SCR_ROOT_DIR}/uv"
    uv venv ${SCR_ROOT_DIR}/uv/vtc --clear
fi

# activate the environment
source ${SCR_ROOT_DIR}/uv/vtc/bin/activate
uv sync
uv pip install -e .