# Best models

This folder contains the best model checkpoints we obtained for the different tasks of this project, as well as associated results.

## 1. [Hover](/models/hover)

For the **Hover** environment, we provide 2 checkpoints:
- `hover-mode0-ppo`: ...
- `hover-mode0-sac`: ...

To evaluate those checkpoints, we use the `evaluate.py` script. The results are presented below.


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
- `waypoint-mode0-ppo`: ...
- `waypoint-mode6-ppo`: ...
- `waypoint-mode6-sac`: ...

To evaluate those checkpoints, we use the `evaluate_norm.py` script. The results are presented below.



| Algo | Mode | Mean reward | Crash rate | Mean waypoints | Mean timesteps |
|------|------|----------------|----------------|----------------|----------------|
| PPO | 0 | 1101.95 ± 364.33 | 5.0 [%] | 3.85 | 285.9 |
| PPO | 6 | 238.06 ± 253.90 | 15.0 [%] | 1.20 | 3070.2 |
| SAC | 6 | -338.27 ± 119.22 | 40.0 [%] | 0.00  | 2691.1 |


## 3. [Dogfight](/models/dogfight)

For the **Dogfight** environment, we provide 1 checkpoint:
- `dogfight`: ...


