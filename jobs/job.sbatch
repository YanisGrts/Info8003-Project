#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=uav        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=outputs/ppo-output.log      # Log-file (important!)
#SBATCH --cpus-per-task=9           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --partition=all
#SBATCH --time=6:00:00              # Max execution time
#

# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/uav

# Run your Python script
python scripts/train_hover.py --flight_mode 7 --steps 1000000 --algo ppo