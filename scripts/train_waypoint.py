import gymnasium
import numpy as np
import PyFlyt.gym_envs
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import os
import argparse
import torch

# Import custom configurations and wrappers for Waypoints
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv

def ppo(flight_mode, run):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Get the specific kwargs for waypoints (flight_dome_size, etc.)
    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = flight_mode

    # 2. Create the vectorized environment with the FlattenWaypointEnv wrapper
    env = make_vec_env(
        "PyFlyt/QuadX-Waypoints-v4",
        n_envs=8, 
        env_kwargs=env_kwargs,
        wrapper_class=FlattenWaypointEnv, # Flattens the Dict observation
        vec_env_cls=SubprocVecEnv,
    )

    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        ent_coef=0.01,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    print(f"Using device: {model.device}")
    return model


def sac(flight_mode, run):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = flight_mode

    env = make_vec_env(
        "PyFlyt/QuadX-Waypoints-v4",
        n_envs=8, 
        env_kwargs=env_kwargs,
        wrapper_class=FlattenWaypointEnv, 
        vec_env_cls=DummyVecEnv, # SAC usually struggles with SubprocVecEnv due to replay buffer constraints
    )
    
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=3e-4,
        buffer_size=500_000,  
        learning_starts=10_000,  
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",         
        device=device,
    )
    print(f"Using device: {model.device}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Drone-Project-Waypoints")
    # Note: For waypoints, flight mode 6 (velocity control) or 7 (position control) are usually easier to start with
    parser.add_argument("--flight_mode", type=int, default=6, choices=[-1,0,4,6,7])
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--steps", type=int, default=1000000) # Increased default steps for navigation
    args = parser.parse_args()
    args.algo = args.algo.lower()

    NAME = f"waypoints-mode{args.flight_mode}-{args.algo}"

    run = wandb.init(
        entity="ChelseaCity",
        project="RL-Drone-Project",
        name=NAME,
        config={
            "environment": "QuadX-Waypoints-v4",
            "algorithm": args.algo,
            "flight_mode": args.flight_mode,
            "total_timesteps": args.steps,
        },
        sync_tensorboard=True, 
        save_code=True,
    )

    if args.algo == "ppo":
        model = ppo(args.flight_mode, run)
    elif args.algo == "sac":
        model = sac(args.flight_mode, run)
    else: 
        raise ValueError("Unknown algo!")

    print(f"Training Waypoints started on Flight Mode {args.flight_mode} with {args.algo.upper()}...")
    
    os.makedirs("models", exist_ok=True)
    
    model.learn(
        total_timesteps=args.steps,
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=1,
        ),
    )
    
    model.save(f"models/{NAME}")
    run.finish()