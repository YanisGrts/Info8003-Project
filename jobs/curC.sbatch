#!/bin/bash

#SBATCH --job-name=sac_runC
#SBATCH --export=ALL
#SBATCH --output=outputs/sac_runC.log
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --partition=tesla

echo "=========================================================="
echo " Run C: SAC Direct Transfer — Full 5M on Dome150/4wp"
echo " Load Phase1 model directly, no intermediate phases"
echo "=========================================================="

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/uav

ALGO="sac"
MODE=6

PHASE1_BASE="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp1"

# ---------------------------------------------------------
# Single phase: 4 Waypoints, 150m Dome — 5M steps
# Goal: let SAC figure it out with correct hyperparams
#       and the full step budget
# ---------------------------------------------------------
echo ">>> Starting Run C (4 Waypoints, 150m Dome, 5M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 1 \
    --num_waypoints 4 --dome_size 150 --steps 5000000 \
    --load_model $PHASE1_BASE \
    --run_id "RunC"

echo "=========================================================="
echo " Run C Completed!"
echo " Final model: models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase1-Dome150-Wp4-RunC"
echo "=========================================================="
