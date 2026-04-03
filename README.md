# INFO8003 — Reinforcement Learning Project

UAV control via deep reinforcement learning using PyFlyt.

## Setup

```bash
pip install -r requirements.txt
```

## Provided scripts

| Script | Purpose |
|--------|---------|
| `scripts/env_config.py` | Environment parameters (waypoint overrides) |
| `scripts/wrappers.py` | `FlattenWaypointEnv` — flattens dict observations |
| `scripts/dogfight_wrapper.py` | `DogfightSelfPlayEnv` — multi-agent → single-agent wrapper |
| `scripts/evaluate.py` | Evaluate a trained model on Hover or Waypoints |
| `scripts/tournament.py` | Elo-rated dogfight tournament |
| `scripts/submission_template.py` | Tournament submission template |

## Evaluation

```bash
# Evaluate a model on hover
python scripts/evaluate.py --model your_model.py --env hover

# Evaluate a model on waypoints
python scripts/evaluate.py --model your_model.py --env waypoints --flight_mode 6

# Run a dogfight tournament
python scripts/tournament.py submissions/
```

## Tournament submission

Copy `scripts/submission_template.py` to `groupXX_name.py` and implement `load_model()`.
Your model must expose: `model.predict(obs, deterministic=True) -> (action, info)`.

See the project statement for full details.
