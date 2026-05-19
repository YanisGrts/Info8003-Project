#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo " Starting Waypoints Curriculum Training Pipeline"
echo "=========================================================="

ALGO="ppo"
MODE=6
STEPS=3000000

# ---------------------------------------------------------
# Phase 1: 1 Target, 25m Dome
# Goal: Teach the agent basic navigation and attitude control
# ---------------------------------------------------------
echo ">>> Starting Phase 1 (1 Target, 25m Dome)..."
python train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 1 \
    --num_waypoints 1 --dome_size 35 --steps $STEPS

PHASE1_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase1-Dome35-Wp1"



# ---------------------------------------------------------
# Phase 2: 3 Targets, 25m Dome
# Goal: Teach the agent to sequence multiple waypoints
# ---------------------------------------------------------
echo ">>> Starting Phase 2 (3 Targets, 25m Dome)..."
python train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 2 \
    --num_waypoints 4 --dome_size 35 --steps $STEPS \
    --load_model $PHASE1_MODEL

PHASE2_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase2-Dome35-Wp4"
STEPS=5000000
# # ---------------------------------------------------------
# # Phase 3: 3 Targets, 125m Dome (Transfer Phase)
# # Goal: Scale up navigation range to the evaluation dome size
# # ---------------------------------------------------------
echo ">>> Starting Phase 3 (3 Targets, 75m Dome)..."
python train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 3 \
    --num_waypoints 4 --dome_size 75 --steps $STEPS \
    --load_model $PHASE2_MODEL

PHASE3_MODEL="models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase3-Dome75-Wp4"
STEPS=10000000

# # ---------------------------------------------------------
# # Phase 4: 4 Targets, 125m Dome (Final Task Difficulty)
# # Goal: Master the full evaluation environment
# # ---------------------------------------------------------
echo ">>> Starting Phase 4 (3 Targets, 125m Dome)..."
python train_waypoint_phase.py \
    --algo $ALGO --flight_mode $MODE --phase 4 \
    --num_waypoints 4 --dome_size 150 --steps $STEPS \
    --load_model $PHASE3_MODEL

# echo ">>> Starting Phase 5 (4 Targets, 125m Dome)..."
# python train_waypoint_phase.py \
#     --algo $ALGO --flight_mode $MODE --phase 5 \
#     --num_waypoints 4 --dome_size 125 --steps $STEPS \
#     --load_model $PHASE3_MODEL

echo "=========================================================="
echo " Curriculum Training Pipeline Completed!"
echo " Final Model saved at: models/waypoint_phase/waypoints-mode${MODE}-${ALGO}-Phase4-Dome125-Wp4"
echo "=========================================================="