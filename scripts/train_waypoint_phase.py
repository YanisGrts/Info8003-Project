import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import os
import argparse
import torch
from stable_baselines3.common.logger import configure
# Import custom configurations and wrappers for Waypoints
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv

class WaypointRewardShaping(gym.Wrapper):
    def __init__(self, env, shaping_coef=0.2):
        super().__init__(env)
        self.shaping_coef = shaping_coef
        self.previous_distance = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._get_distance(obs)
        return obs, info

    def _get_distance(self, obs):
        if isinstance(obs, dict) and "target_deltas" in obs:
            targets = obs["target_deltas"]
            if len(targets) > 0:
                return float(np.linalg.norm(targets[0]))
        return 0.0

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        current_distance = self._get_distance(obs)

        if base_reward >= 10.0 or base_reward <= -10.0:
            self.previous_distance = current_distance
            return obs, base_reward, terminated, truncated, info

        current_potential = -self.shaping_coef * current_distance
        previous_potential = -self.shaping_coef * self.previous_distance

        shaping = current_potential - previous_potential
        self.previous_distance = current_distance

        time_penalty = -0.1

        raw_yaw_rate = self.env.unwrapped.env.state(0)[0][2]
        yaw_penalty = -0.01 * (raw_yaw_rate ** 2)

        custom_reward = shaping + time_penalty + yaw_penalty

        return obs, custom_reward, terminated, truncated, info

def make_custom_env(env_id, env_kwargs, rank, seed=0):
    """Utility function to chain multiple wrappers for a multiprocessed env."""
    def _init():
        # BAse
        env = gym.make(env_id, **env_kwargs)
        # Custom Reward
        env = WaypointRewardShaping(env)
        env = FlattenWaypointEnv(env, max_waypoints=4)
        env.reset(seed=seed + rank)
        return env
    return _init

def ppo(args, run):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = args.flight_mode
    
    # Apply the dome size to PyFlyt 
    env_kwargs["flight_dome_size"] = args.dome_size
    env_kwargs["num_targets"] = args.num_waypoints

    env = SubprocVecEnv([
        make_custom_env("PyFlyt/QuadX-Waypoints-v4", env_kwargs, i) 
        for i in range(8)
    ])

    env = VecMonitor(env)

    if args.load_model is not None:
        print(f"Loading previous model and normalization stats from: {args.load_model}")
        
        # Load the normalization stats 
        vec_norm_path = f"{args.load_model}_vecnormalize.pkl"
        env = VecNormalize.load(vec_norm_path, env)
        
        # Load the PPO model
        custom_objects = {
            "learning_rate": 3e-5, # Drop it from 1e-4 to 3e-5
            "target_kl": 0.015
        }
        model = PPO.load(args.load_model, env=env, device=device, custom_objects=custom_objects)
        
        # Set up the new logger for this specific Phase
        new_logger = configure(f"runs/{run.id}", ["csv", "tensorboard"])
        model.set_logger(new_logger)
        
    else:
        print("Initializing completely new PPO model...")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)#True)
        # model = PPO(
        #     "MlpPolicy",
        #     env,
        #     verbose=0,
        #     tensorboard_log=f"runs/{run.id}",
        #     learning_rate=1e-4,
        #     n_steps=2048,
        #     batch_size=256,
        #     ent_coef=0.01,
        #     gae_lambda=0.88,
        #     clip_range=0.2,
        #     policy_kwargs=dict(net_arch=[256, 256, 256]),
        #     device=device,
        # )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=f"runs/{run.id}",
            learning_rate=3e-4,     # Standard starting LR
            n_steps=4096,           # Increased horizon
            batch_size=256,
            ent_coef=0.01,
            gae_lambda=0.95,        # Restored to 0.95 for better long-term credit assignment
            clip_range=0.2,
            use_sde=True,           # ADDED SDE for smoother flight
            policy_kwargs=dict(
                net_arch=[256, 256], 
                log_std_init=-2     # Good starting point for SDE
            ),
            device=device,
        )
        
    print(f"Using device: {model.device}")
    return model, env

def sac(flight_mode, run):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = flight_mode

    # Create the vectorized environment using the custom builder
    env = SubprocVecEnv([
        make_custom_env("PyFlyt/QuadX-Waypoints-v4", env_kwargs, i) 
        for i in range(8)
    ])

    # Apply the standard SB3 vector wrappers
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Best configuration from tuning
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        device=device,
        learning_rate=0.0004603865150666861,
        buffer_size=1_000_000,  
        learning_starts=2000,  
        batch_size=128,
        tau=0.013409850247145992,
        gamma=0.9829025672846582,
        train_freq=1,
        gradient_steps=-1,
        ent_coef="auto",
        target_entropy=-12.597293711066428
    )
    return model, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Drone-Project-Waypoints")
    # Note: For waypoints, flight mode 6 (velocity control) or 7 (position control) are usually easier to start with
    parser.add_argument("--flight_mode", type=int, default=6, choices=[-1,0,4,6,7])
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--steps", type=int, default=1000000) # Increased default steps for navigation

    parser.add_argument("--dome_size", type=float, default=20.0, help="Radius of the waypoint spawn dome")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a previously trained model (.zip)")
    parser.add_argument("--phase", type=int, default=1, help="Phase number for naming the run")
    parser.add_argument("--num_waypoints", type=int, default=1, help="Number of active targets")

    args = parser.parse_args()
    args.algo = args.algo.lower()

    NAME = f"waypoints-mode{args.flight_mode}-{args.algo}-Phase{args.phase}-Dome{int(args.dome_size)}-Wp{args.num_waypoints}"

    run = wandb.init(
        entity="ChelseaCity",
        project="RL-Drone-Project",
        name=NAME,
        config={
            "environment": "QuadX-Waypoints-v4",
            "algorithm": args.algo,
            "flight_mode": args.flight_mode,
            "total_timesteps": args.steps,
            "dome_size": args.dome_size,
            "phase": args.phase,
        },
        sync_tensorboard=True, 
        save_code=True,
    )

    if args.algo == "ppo":
        model, env = ppo(args, run)
    elif args.algo == "sac":
        model, env = sac(args.flight_mode, run)
    else: 
        raise ValueError("Unknown algo!")

    print(f"Training Waypoints started on Flight Mode {args.flight_mode} with {args.algo.upper()}...")
    
    os.makedirs("models", exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // 8, 1),  # every ~100k env steps, adjusted for num_envs
        save_path=f"models/waypoint_phase/{NAME}_checkpoints/",
        name_prefix=NAME,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([
            checkpoint_callback,
            WandbCallback(
                verbose=1,
            ),
        ]),
    )
        
    model.save(f"models/waypoint_phase/{NAME}")
    env.save(f"models/waypoint_phase/{NAME}_vecnormalize.pkl")
    run.finish()