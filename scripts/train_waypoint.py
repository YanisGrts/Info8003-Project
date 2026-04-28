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

# Import custom configurations and wrappers for Waypoints
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv
    

class WaypointRewardShaping(gym.Wrapper):
    def __init__(self, env, gamma=0.95):
        super().__init__(env)
        self.gamma = gamma
        self.previous_distance = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_waypoints = 0
        self.previous_distance = self._get_distance(obs)
        return obs, info

    def _get_distance(self, obs):
        if isinstance(obs, dict) and "target_deltas" in obs:
            targets = obs["target_deltas"]
            if len(targets) > 0:
                return float(np.linalg.norm(targets[0]))
        return 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_distance = self._get_distance(obs)

        # Potential-based shaping
        shaping = self.gamma * (-current_distance) - (-self.previous_distance)
        self.previous_distance = current_distance

        # Capture bonus on actual waypoint capture
        if info.get("num_targets_reached", 0) > self._last_waypoints:
            shaping += 5.0
        self._last_waypoints = info.get("num_targets_reached", 0)

        return obs, reward + shaping, terminated, truncated, info
    # def step(self, action):
    #     obs, reward, terminated, truncated, info = self.env.step(action)
        
    #     # 1. Base reward (Keep the crash penalty)
    #     shaped_reward = reward 

    #     # 2. BRUTAL Time Penalty
    #     # -0.1 was too soft. -0.5 makes hovering intolerable.
    #     shaped_reward -= 0.2

    #     if self.previous_action is not None:
    #         action_diff = np.linalg.norm(action - self.previous_action)
    #         shaped_reward -= 0.1 * action_diff # Penalize jerky movements
        
    #     self.previous_action = np.array(action)
    #     if isinstance(obs, dict) and "target_deltas" in obs:
    #         targets = obs["target_deltas"]
    #         if len(targets) > 0:
    #             current_distance = np.linalg.norm(targets[0])

    #             if self.previous_distance is not None:
    #                 # 3. MASSIVE Progress Reward
    #                 # We need to "pull" the agent out of its hover state.
    #                 progress = self.previous_distance - current_distance
    #                 shaped_reward += 2.0 * progress  # Increased from 2.0 to 10.0

    #             # 4. Stronger Magnet Bonus
    #             shaped_reward += 1.0 * np.exp(-0.1 * current_distance)
    #             self.previous_distance = current_distance

    #     return obs, shaped_reward, terminated, truncated, info
    # def step(self, action):
    #     obs, reward, terminated, truncated, info = self.env.step(action)
    #     shaped_reward = reward 

    #     # 1. BRUTAL Time Penalty
    #     # -0.2 was not enough to scare it. -0.5 makes the agent "panic."
    #     shaped_reward -= 0.5 

    #     if isinstance(obs, dict) and "target_deltas" in obs:
    #         targets = obs["target_deltas"]
    #         if len(targets) > 0:
    #             current_distance = np.linalg.norm(targets[0])

    #             if self.previous_distance is not None:
    #                 # 2. MASSIVE Progress Reward
    #                 # We are increasing this to 20.0. 
    #                 # This means moving 1 meter closer is 40x better than staying still.
    #                 progress = self.previous_distance - current_distance
    #                 shaped_reward += 20.0 * progress 

    #             # 3. The "Capture" Bonus
    #             # Only give this when the agent is actually "touching" the target.
    #             if current_distance < 4.0:
    #                 shaped_reward += 10.0  # Huge bonus for being inside the target zone

    #             self.previous_distance = current_distance

    #     return obs, shaped_reward, terminated, truncated, info
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
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=512,
        ent_coef=0.0005, #0.01,
        gae_lambda=0.85,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    print(f"Using device: {model.device}")
    return model, env


def sac(flight_mode, run):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
    args = parser.parse_args()
    args.algo = args.algo.lower()

    NAME = f"waypoints-mode{args.flight_mode}-{args.algo}-tuned2"

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
        model, env = ppo(args.flight_mode, run)
    elif args.algo == "sac":
        model, env = sac(args.flight_mode, run)
    else: 
        raise ValueError("Unknown algo!")

    print(f"Training Waypoints started on Flight Mode {args.flight_mode} with {args.algo.upper()}...")
    
    os.makedirs("models", exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // 8, 1),  # every ~100k env steps, adjusted for num_envs
        save_path=f"models/waypoint/{NAME}_checkpoints/",
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
        
    model.save(f"models/waypoint/{NAME}")
    env.save(f"models/waypoint/{NAME}_vecnormalize.pkl")
    run.finish()