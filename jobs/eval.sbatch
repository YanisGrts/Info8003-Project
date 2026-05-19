#!/bin/bash

#SBATCH --job-name=sac_eval_all
#SBATCH --export=ALL
#SBATCH --output=outputs/sac_eval_all.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:0
#SBATCH --time=8:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/uav

# ── Configuration ─────────────────────────────────────────────
MODE=6
N_EPISODES=10
ALGO="sac"

# ── Phase configurations: "ModelName DomeSize NumTargets" ─────
ORIGINAL=(
    "waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp1 25 1"
    "waypoints-mode${MODE}-${ALGO}-Phase2-Dome25-Wp4-BIS 25 4"
    "waypoints-mode${MODE}-${ALGO}-Phase3-Dome75-Wp4-BIS 75 4"
)

RUN_A=(
    "waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp2-RunA 25 2"
    "waypoints-mode${MODE}-${ALGO}-Phase2-Dome75-Wp4-RunA 75 4"
    "waypoints-mode${MODE}-${ALGO}-Phase3-Dome150-Wp4-RunA 150 4"
)

RUN_B=(
    "waypoints-mode${MODE}-${ALGO}-Phase1-Dome25-Wp4-RunB 25 4"
    "waypoints-mode${MODE}-${ALGO}-Phase2-Dome150-Wp4-RunB 150 4"
)

# ── Helper function ───────────────────────────────────────────
evaluate_phases() {
    local CURRICULUM_NAME=$1
    shift
    local PHASES=("$@")

    echo "=========================================================="
    echo " Evaluating Curriculum: $CURRICULUM_NAME"
    echo "=========================================================="

    for CONFIG in "${PHASES[@]}"; do
        read -r MODEL_NAME DOME TARGETS <<< "$CONFIG"

        MODEL_PATH="models/waypoint_phase/${MODEL_NAME}.zip"
        VECNORM_PATH="models/waypoint_phase/${MODEL_NAME}_vecnormalize.pkl"

        if [ ! -f "$MODEL_PATH" ]; then
            echo "[WARNING] Could not find $MODEL_PATH. Skipping..."
            continue
        fi

        echo ">>> Evaluating: $MODEL_NAME"
        echo ">>> Parameters: Dome=$DOME, Targets=$TARGETS"

        python scripts/evaluate_norm.py \
            --model "$MODEL_PATH" \
            --vecnorm "$VECNORM_PATH" \
            --env waypoints \
            --flight_mode "$MODE" \
            --n_episodes "$N_EPISODES" \
            --dome_size "$DOME" \
            --num_targets "$TARGETS"

        echo ""
    done
}

# ── Run all evaluations ───────────────────────────────────────
evaluate_phases "Original" "${ORIGINAL[@]}"
evaluate_phases "Run A"    "${RUN_A[@]}"
evaluate_phases "Run B"    "${RUN_B[@]}"

echo "=========================================================="
echo " All Curriculum Evaluations Completed!"
echo "=========================================================="