# Evaluation of the models

## 1. Hover

Evaluation of both algorithms on the five environments using:
```bash
python scripts/evaluate.py --model models/hover-modeX-<algo>.zip --env hover --flight_mode X
```

**Results**:
| Mode | PPO mean reward | PPO crash rate | SAC mean reward | SAC crash rate |
|------|----------------|----------------|----------------|----------------|
| 7    | +873.7 ± 0.2   | `0%`             | +1102.1 ± 0.9  | `0%`             |
| 6    | −706.2 ± 33.2  | 35%            | +380.1 ± 23.5  | `0%`             |
| 4    | +196.7 ± 140.8 | 30%            | +516.8 ± 77.0  | `0%`            |
| 0    | −73.0 ± 0.1    | 100%           | −85.9 ± 0.2    | 100%           |
| −1   | −476.3 ± 242.9 | 95%            | −119.5 ± 129.2 | 95%            |

## 2. Waypoints

Evaluation of both algorithms on the five environments using:
```bash
python scripts/evaluate_norm.py \
  --model models/waypoints-modeX-<algo>.zip \
  --env waypoints \
  --flight_mode X \
  --norm_path models/waypoints-modeX-<algo>_vecnormalize.pkl
```

| Mode | PPO mean reward | PPO crash rate | SAC mean reward | SAC crash rate |
|------|----------------|----------------|----------------|----------------|
| 7 | -106.85 ± 136.20 | `0%` | -72.74 ± 113.71 | `0%` |
| 6 | -13.61 ± 117.19 | 25% | ... | ... |
| 4 | ... | ... | ... | ... |
| 0 | ... | ... | ... | ... |
| -1 | ... | ... | ... | ... |