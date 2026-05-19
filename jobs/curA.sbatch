#!/bin/bash

#SBATCH --job-name=sac_runA
#SBATCH --export=ALL
#SBATCH --output=outputs/sac_runA.log
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --partition=1080ti

echo "=========================================================="
echo " Run A: SAC Tight 3-Phase Curriculum"
echo " Phase1(25/2wp/1.5M) -> Phase2(75/4wp/1.5M) -> Phase3(150/4wp/2M)"
echo "=========================================================="

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/uav

ALGO="sac"
MODE=6

PHASE1_BASE="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp1"

# ---------------------------------------------------------
# Phase A1: 2 Waypoints, 25m Dome — 1.5M steps
# Goal: consolidate multi-waypoint nav in familiar space
# Load from Phase1 (DO NOT reset obs_rms — handled in script)
# ---------------------------------------------------------
echo ">>> Starting Phase A1 (2 Waypoints, 25m Dome, 1.5M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 1 \
    --num_waypoints 2 --dome_size 25 --steps 1500000 \
    --load_model $PHASE1_BASE \
    --run_id "RunA"

PHASE_A1_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp2-RunA"

# ---------------------------------------------------------
# Phase A2: 4 Waypoints, 75m Dome — 1.5M steps
# Goal: scale up waypoints and dome together
# ---------------------------------------------------------
echo ">>> Starting Phase A2 (4 Waypoints, 75m Dome, 1.5M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 2 \
    --num_waypoints 4 --dome_size 75 --steps 1500000 \
    --load_model $PHASE_A1_MODEL \
    --run_id "RunA"

PHASE_A2_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase2-Dome75-Wp4-RunA"

# ---------------------------------------------------------
# Phase A3: 4 Waypoints, 150m Dome — 2M steps
# Goal: master the final evaluation environment
# ---------------------------------------------------------
echo ">>> Starting Phase A3 (4 Waypoints, 150m Dome, 2M steps)..."
python scripts/train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 3 \
    --num_waypoints 4 --dome_size 150 --steps 2000000 \
    --load_model $PHASE_A2_MODEL \
    --run_id "RunA"

echo "=========================================================="
echo " Run A Completed!"
echo " Final model: models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase3-Dome150-Wp4-RunA"
echo "=========================================================="
