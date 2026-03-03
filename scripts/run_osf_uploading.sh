#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-output/osf_uploading_%j.out
#SBATCH --error=slurm-output/osf_uploading_%j.err

source ~/.zshrc
cd ~/vlm-tg-context

conda activate r-env

Rscript upload_osf.R