#!/bin/bash

# Array of configurations to test: "ModelName DomeSize NumTargets"
MODE=0
PHASES=(
    "waypoints-mode${MODE}-ppo-Phase1-Dome25-Wp1 25 1"
    "waypoints-mode${MODE}-ppo-Phase2-Dome25-Wp4 25 4"
    "waypoints-mode${MODE}-ppo-Phase3-Dome75-Wp4 75 4"
    "waypoints-mode${MODE}-ppo-Phase4-Dome150-Wp4 150 4"
)

echo "=========================================================="
echo " Starting Curriculum Evaluation"
echo "=========================================================="

for CONFIG in "${PHASES[@]}"; do
    # Extract variables from the string
    read -r MODEL_NAME DOME TARGETS <<< "$CONFIG"
    
    MODEL_PATH="models/waypoint_phase/${MODEL_NAME}.zip"
    VECNORM_PATH="models/waypoint_phase/${MODEL_NAME}_vecnormalize.pkl"

    # Check if the model actually exists before trying to evaluate
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[WARNING] Could not find $MODEL_PATH. Skipping..."
        continue
    fi

    echo ">>> Evaluating Phase: $MODEL_NAME"
    echo ">>> Parameters: Dome=$DOME, Targets=$TARGETS"
    
    python evaluate_norm.py \
        --model "$MODEL_PATH" \
        --vecnorm "$VECNORM_PATH" \
        --env waypoints \
        --flight_mode "$MODE" \
        --n_episodes 10 \
        --dome_size "$DOME" \
        --num_targets "$TARGETS"
    
    echo ""
done

echo "=========================================================="
echo " All Phase Evaluations Completed!"
echo "=========================================================="