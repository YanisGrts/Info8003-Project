#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=eval-uav        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=outputs/eval-output.log      # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --partition=all
#SBATCH --time=1:00:00              # Max execution time
#

# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/uav

# Run your Python script
python scripts/evaluate.py --model models/hover-mode7-sac.zip --env hover