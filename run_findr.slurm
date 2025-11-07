#!/bin/bash
#SBATCH --job-name=findr_training
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64G
#SBATCH --output=findr_%j.out
#SBATCH --error=findr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shunli@g.harvard.edu

# =============================================================================
# Edit the variables below
# =============================================================================

# Configuration
findr_dir="/n/holylabs/LABS/bsabatini_lab/Users/shunnnli/findr"
conda_env="findr"

# Set datapath and analysispath
datapath="/n/holylabs/LABS/bsabatini_lab/Users/shunnnli/findr/datasets/20250622-Data-InsCtx/20250618_M346_4W50_g0_ccm_findr_dataset.npz"
analysispath="/n/holylabs/LABS/bsabatini_lab/Users/shunnnli/findr/datasets/20250622-Data-InsCtx/findr-results-20251107"

# Load modules
module load python/3.10.9-fasrc01

# Activate conda environment
# source ~/.bashrc
conda activate $conda_env

# Create output directory
mkdir -p "$analysispath"

# Run FINDR
cd "$findr_dir"
python main.py --datapath="$datapath" --workdir="$analysispath"

