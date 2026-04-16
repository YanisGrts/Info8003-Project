import gymnasium
import PyFlyt.gym_envs
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import os
import argparse
import torch

def ppo(flight_mode, run):
    # env = gymnasium.make("PyFlyt/QuadX-Hover-v4", flight_mode=0)
    env = make_vec_env(
        "PyFlyt/QuadX-Hover-v4",
        n_envs=8,
        env_kwargs={"flight_mode": flight_mode},
        vec_env_cls=SubprocVecEnv,
    )
    env = VecMonitor(env) # Necessary for Wandb to track episode rewards

    assert torch.cuda.is_available(), "CUDA not available!"

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.01,   # add this — helps exploration early on
        device="cuda",
    )
    print(f"Using device: {model.device}")

    return model

def sac(flight_mode, run):
    env = gymnasium.make("PyFlyt/QuadX-Hover-v4", flight_mode=flight_mode)
    env = Monitor(env)

    assert torch.cuda.is_available(), "CUDA not available!"

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,  # collect random experience before updating
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",         # SAC auto-tunes entropy for exploration
        device="cuda",
    )
    print(f"Using device: {model.device}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Drone-Project")
    parser.add_argument("--flight_mode", type=int, default=7, choices=[-1,0,4,6,7])
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()
    args.algo = args.algo.lower()

    NAME = f"hover-mode{args.flight_mode}-{args.algo}"

    run = wandb.init(
        entity="ChelseaCity",
        project="RL-Drone-Project",
        name=NAME,
        config={
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

    print(f"Training started on Flight Mode {args.flight_mode}...")
    model.learn(
        total_timesteps=args.steps,
        callback=WandbCallback(
            model_save_path=f"models/{run.id}", # need to have a 'models' folder
            verbose=0,
        ),
    )

    model.save(f"models/{NAME}")
    run.finish()