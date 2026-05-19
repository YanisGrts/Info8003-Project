#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=sac_1        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=outputs/sac_stage1_bis.log      # Log-file (important!)
#SBATCH --cpus-per-task=9         # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=24:00:00              # Max execution time
#

# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/uav

# Run your Python script
python scripts/train_waypoint_phase.py \
    --flight_mode 6 \
    --algo "sac" \
    --steps 200000 \
    --num_waypoints 1 \
    --dome_size 30.0 \
    --phase 1