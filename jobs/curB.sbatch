#!/bin/bash

#SBATCH --job-name=sac_runB
#SBATCH --export=ALL
#SBATCH --output=outputs/sac_runB.log
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --partition=quadro

echo "=========================================================="
echo " Run B: SAC Aggressive 2-Phase — Max time on final env"
echo " Phase1(25/4wp/1.5M) -> Phase2(150/4wp/3.5M)"
echo "=========================================================="

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/uav

ALGO="sac"
MODE=6

PHASE1_BASE="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp1"

# ---------------------------------------------------------
# Phase B1: 4 Waypoints, 25m Dome — 1.5M steps
# Goal: force 4-waypoint learning in small dome quickly
# ---------------------------------------------------------
echo ">>> Starting Phase B1 (4 Waypoints, 25m Dome, 1.5M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 1 \
    --num_waypoints 4 --dome_size 25 --steps 1500000 \
    --load_model $PHASE1_BASE \
    --run_id "RunB"

PHASE_B1_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp4-RunB"

# ---------------------------------------------------------
# Phase B2: 4 Waypoints, 150m Dome — 3.5M steps
# Goal: spend the bulk of the budget on the final env
# ---------------------------------------------------------
echo ">>> Starting Phase B2 (4 Waypoints, 150m Dome, 3.5M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 2 \
    --num_waypoints 4 --dome_size 150 --steps 3500000 \
    --load_model $PHASE_B1_MODEL \
    --run_id "RunB"

echo "=========================================================="
echo " Run B Completed!"
echo " Final model: models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase2-Dome150-Wp4-RunB"
echo "=========================================================="
