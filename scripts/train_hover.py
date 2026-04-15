import gymnasium
import PyFlyt.gym_envs
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Remove the Monitor wrapper — VecEnv handles that internally
import os
import torch

# Create folders for saving models
os.makedirs("./models/", exist_ok=True)

# 2. Initialize Wandb
run = wandb.init(
    project="RL-Drone-Project",
    name="hover-mode7-ppo",
    config={
        "algorithm": "PPO",
        "flight_mode": 7,  # Easiest mode: Position Control
        "total_timesteps": 200000,
    },
    sync_tensorboard=True, 
    save_code=True,
)

# 3. Setup Env
# Using Flight Mode 7 as your first test
env = make_vec_env(
    "PyFlyt/QuadX-Hover-v4",
    n_envs=8,
    env_kwargs={"flight_mode": 7},
    vec_env_cls=SubprocVecEnv,  # <-- real parallelism
)
env = Monitor(env) # Necessary for Wandb to track episode rewards


# 4. Setup Model
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=f"runs/{run.id}",
    learning_rate=3e-4,
    device="cuda",
)
print(f"Using device: {model.device}")
assert torch.cuda.is_available(), "CUDA not available!"

# 5. Train
print("Training started on Flight Mode 7...")
model.learn(
    total_timesteps=run.config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=0,
    ),
)

# 6. Save final
model.save(f"models/hover_mode7_final")
run.finish()