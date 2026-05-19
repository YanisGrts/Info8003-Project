# Best models

This folder contains the best model checkpoints we obtained for the different tasks of this project, as well as associated results.

## 1. [Hover](/models/hover)

For the **Hover** environment, we provide 6 checkpoints (4 for PPO, 2 for SAC). They were all trained with `train_hover.py`and evaluated with `evaluate.py`. The results are presented below.


### Results - Flight mode 0
| Algorithm | Flight mode | Mean reward | Crash rate |
|------|----------------|----------------|----------------|
| PPO | 0 | 1137.99 ± 1.59 | 0 [%] |
| SAC | 0 | 1138.33 ± 0.08 | 0 [%] |

### Flight mode comparison
| Flight mode | SAC Timesteps | PPO Timesteps |
|------|----------------|----------------|
| -1 | *DNR* | *DNR* |
| 0  | 450,000 | 1,500,000 |
| 4  | 350,000 | 2,000,000 |
| 6  | 250,000 | 1,100,000 |
| 7  | 110,000 | 1,100,000 |

`NB:` *DNR* = *Did Not Reach* the threshold.


## 2. [Waypoints](/models/waypoint)

For the **Waypoints** environment, we provide 3 checkpoints (each with a `.pkl` file storing the normalizations):
- `waypoints-mode0-ppo-Phase4-Dome150-Wp4`: PPO model trained with `train_waypoint_phase.py` and evaluated with `evaluate_norm.py`.
- `waypoints-simple-mode6-ppo-Phase2-Dome150-Wp4`: PPO model trained with `train_waypoint_phase_simple.py` and evaluated with `evaluate_simple_obs.py`.
- `waypoints-mode6-sac-Phase3-Dome150-Wp4-RunA`: SAC model trained with `train_waypoint_phase.py` and evaluated with `evaluate_norm.py`.

The results for those three checkpoints are presented below.



| Algo | Mode | Mean reward | Crash rate | Mean waypoints | Mean timesteps |
|------|------|----------------|----------------|----------------|----------------|
| PPO | 0 | 1101.95 ± 364.33 | 5.0 [%] | 3.85 | 285.9 |
| PPO | 6 | 238.06 ± 253.90 | 15.0 [%] | 1.20 | 3070.2 |
| SAC | 6 | -338.27 ± 119.22 | 40.0 [%] | 0.00  | 2691.1 |


## 3. [Dogfight](/models/dogfight)

For the **Dogfight** environment, we provide 4 checkpoint:
- `dogfight-phase1-sac`: SAC model baseline trained with `train_dogfight`. 
- `dogfight-selfplay-phase1`,`dogfight-selfplay-phase2`, and `dogfight-selfplay-phase3`: PPO models trained with `train_dogfight`. Each version was trained starting from the previous phase (from scratch for the first phase)

To select the best model for the final submission, we evaluated all checkpoints in a round-robin tournament (10 matches per pair). The models are ranked below based on their final Elo scores:

| Rank | Model Checkpoint | Final Elo |
| :--- | :--- | :--- |
| 1 | `dogfight-selfplay-phase1` | **1528.0** |
| 2 | `dogfight-selfplay-phase3` | 1509.0 |
| 3 | `dogfight-selfplay-phase2` | 1487.4 |
| 4 | `dogfight-phase1-sac` | 1475.6 |
