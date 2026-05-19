#!/bin/bash

#SBATCH --job-name=sac_curriculum
#SBATCH --export=ALL
#SBATCH --output=outputs/sac_curriculum.log
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:0
#SBATCH --time=72:00:00

echo "=========================================================="
echo " Starting SAC Waypoints Curriculum Training Pipeline"
echo "=========================================================="

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/uav

ALGO="sac"
MODE=6

# ---------------------------------------------------------
# Phase 1: 1 Waypoint, 25m Dome — 3M steps
# Goal: Teach basic navigation in a small space
# ---------------------------------------------------------
echo ">>> Starting Phase 1 (1 Waypoint, 25m Dome, 3M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 1 \
    --num_waypoints 1 --dome_size 25 --steps 3000000

PHASE1_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp1"

# ---------------------------------------------------------
# Phase 2: 4 Waypoints, 25m Dome — 3M steps
# Goal: Teach waypoint sequencing in a familiar space
# ---------------------------------------------------------
echo ">>> Starting Phase 2 (4 Waypoints, 25m Dome, 3M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 2 \
    --num_waypoints 4 --dome_size 25 --steps 3000000 \
    --load_model $PHASE1_MODEL

PHASE2_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase2-Dome25-Wp4"

# ---------------------------------------------------------
# Phase 3: 4 Waypoints, 75m Dome — 5M steps
# Goal: Transfer navigation to a larger space
# ---------------------------------------------------------
echo ">>> Starting Phase 3 (4 Waypoints, 75m Dome, 5M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 3 \
    --num_waypoints 4 --dome_size 75 --steps 5000000 \
    --load_model $PHASE2_MODEL

PHASE3_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase3-Dome75-Wp4"

# ---------------------------------------------------------
# Phase 4: 4 Waypoints, 150m Dome — 10M steps
# Goal: Master the full evaluation environment
# ---------------------------------------------------------
echo ">>> Starting Phase 4 (4 Waypoints, 150m Dome, 10M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 4 \
    --num_waypoints 4 --dome_size 150 --steps 10000000 \
    --load_model $PHASE3_MODEL

echo "=========================================================="
echo " SAC Curriculum Training Pipeline Completed!"
echo " Final model: models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase4-Dome150-Wp4"
echo "=========================================================="