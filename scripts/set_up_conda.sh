# install miniconda if not already installed
if [ ! -d "/scr/benpry/conda" ]; then
    echo "Conda not found at /scr/benpry/conda. Installing Miniconda..."
    mkdir -p /scr/benpry
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /scr/benpry/miniforge.sh
    bash /scr/benpry/miniforge.sh -b -p /scr/benpry/conda
    rm /scr/benpry/miniforge.sh
else
    echo "Conda installation found at /scr/benpry/conda."
fi

# check if there is a conda environment called vtc
if [ ! -d "/scr/benpry/conda/envs/vtc" ]; then
    echo "Conda environment vtc not found. Creating..."
    conda env create -f environment.yml -y
    conda activate vtc
    pip install -e .
else
    echo "Conda environment vtc found."
    conda activate vtc
fi