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

import gymnasium as gym
import numpy as np

import gymnasium as gym
import numpy as np

class WaypointRewardShaping(gym.Wrapper):
    """Custom wrapper to shape the reward for the Waypoints environment."""
    
    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Initialize the previous distance at the start of the episode
        if isinstance(obs, dict) and "target_deltas" in obs:
            targets = obs["target_deltas"]
            if len(targets) > 0:
                self.previous_distance = np.linalg.norm(targets[0])
            else:
                self.previous_distance = 0.0
                
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = reward * 10

        # Inciter à aller vite 
        shaped_reward -= 0.01
        
        #  Limiter les gros changements d'action
        action_penalty = -0.001 * np.sum(np.square(action)) 
        shaped_reward += action_penalty
        
        #Récompenser le fait de se rapprocher
        if isinstance(obs, dict) and "target_deltas" in obs:
            targets = obs["target_deltas"]
            if len(targets) > 0:
                current_distance = np.linalg.norm(targets[0])
                
                if self.previous_distance is not None:
                    progress = self.previous_distance - current_distance
                    distance_reward = 100.0 * progress 
                    shaped_reward += distance_reward
                
                self.previous_distance = current_distance
                
        return obs, shaped_reward, terminated, truncated, info

def make_custom_env(env_id, env_kwargs, rank, seed=0):
    """Utility function to chain multiple wrappers for a multiprocessed env."""
    def _init():
        # 1. Create the base environment
        env = gymnasium.make(env_id, **env_kwargs)
        
        # 2. Add your custom reward shaping FIRST 
        # (It needs to be first so it can read the 'target_deltas' dictionary before it gets flattened)
        env = WaypointRewardShaping(env) 
        
        # 3. Flatten the observation LAST
        # (Must be last so the PPO/SAC algorithms receive the 1D vector they expect)
        env = FlattenWaypointEnv(env, max_waypoints=4)
        
        env.reset(seed=seed + rank)
        return env
    return _init

def ppo(flight_mode, run):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = flight_mode

    # 2. Create the vectorized environment using the custom builder
    # We create a list of 8 independent environments using a list comprehension
    env = SubprocVecEnv([
        make_custom_env("PyFlyt/QuadX-Waypoints-v4", env_kwargs, i) 
        for i in range(8)
    ])

    # 3. Apply the standard SB3 vector wrappers
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.05,
        gae_lambda=0.95,
        clip_range=0.15,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
        device=device,
    )

    print(f"Using device: {model.device}")
    return model


def sac(flight_mode, run):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = flight_mode

    # 2. Create the vectorized environment using the custom builder
    # We create a list of 8 independent environments using a list comprehension
    env = SubprocVecEnv([
        make_custom_env("PyFlyt/QuadX-Waypoints-v4", env_kwargs, i) 
        for i in range(8)
    ])

    # 3. Apply the standard SB3 vector wrappers
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=1e-4,
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