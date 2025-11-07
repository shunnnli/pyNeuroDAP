#!/bin/bash
#SBATCH -p kempner # partition (queue)
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --job-name=findr-training     # create a short name for your job
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --cpus-per-task=16       # number of cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G               # total memory per node (16 GB per cpu-core is default)
#SBATCH --time=3-00:00:00        # total run time limit (HH:MM:SS)
#SBATCH -o %j.out # STDOUT
#SBATCH -e %j.err # STDERR
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user shunli@g.harvard.edu

# Configuration
findr_dir="/n/holylabs/LABS/bsabatini_lab/Users/shunnnli/findr"
mamba_env="findr"

# Set datapath and analysispath (EDIT THIS!!)
datapath="/n/holylabs/LABS/bsabatini_lab/Users/shunnnli/findr/datasets/20250622-Data-InsCtx/20250618_M346_4W50_g0_ccm_findr_dataset.npz"
analysispath="/n/holylabs/LABS/bsabatini_lab/Users/shunnnli/findr/datasets/20250622-Data-InsCtx/findr-results-20251107"

# Load modules
module load python/3.10.9-fasrc01
module load Mambaforge/23.11.0-fasrc01

# Activate environment
# source ~/.bashrc
mamba activate $mamba_env

# Create output directory
mkdir -p "$analysispath"

# Run FINDR
cd "$findr_dir"
python main.py --datapath="$datapath" --workdir="$analysispath"

# Move job outputs to experiment directory
mv $SLURM_JOB_ID.out $analysispath
mv $SLURM_JOB_ID.err $analysispath
