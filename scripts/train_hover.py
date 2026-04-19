import gymnasium
import numpy as np
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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize


def ppo(flight_mode, run):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Use 8 environments (or more, depending on your CPU)
    env = make_vec_env(
        "PyFlyt/QuadX-Hover-v4",
        n_envs=8, 
        env_kwargs={"flight_mode": flight_mode},
        vec_env_cls=SubprocVecEnv,  # Use multiprocessing for high FPS
    )

    env = VecMonitor(env)
    # KEEP VecNormalize! This is why your brown run succeeded.
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
    env = make_vec_env(
        "PyFlyt/QuadX-Hover-v4",
        n_envs=8, 
        env_kwargs={"flight_mode": flight_mode},
        vec_env_cls=DummyVecEnv,
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps" 
    else:
        device = "cpu"

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

# def sac(flight_mode, run):
#     env = gymnasium.make("PyFlyt/QuadX-Hover-v4", flight_mode=flight_mode)
#     env = Monitor(env)

#     if torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"

#     model = SAC(
#         "MlpPolicy",
#         env,
#         verbose=0,
#         tensorboard_log=f"runs/{run.id}",
#         learning_rate=3e-4,
#         buffer_size=300_000,
#         learning_starts=10_000,  # collect random experience before updating
#         batch_size=256,
#         tau=0.005,
#         gamma=0.99,
#         ent_coef="auto",         # SAC auto-tunes entropy for exploration
#         device=device,
#     )
#     print(f"Using device: {model.device}")

#     return model

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
            verbose=1,
        ),
    )
    
    model.save(f"models/{NAME}")
    run.finish()