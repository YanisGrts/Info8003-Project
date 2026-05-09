import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
import os
import argparse
import torch

# Import custom configurations and wrappers for Waypoints
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv

class WaypointMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_waypoints = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for info, done in zip(infos, dones):
            if done and "num_targets_reached" in info:
                self.episode_waypoints.append(info["num_targets_reached"])

        if len(self.episode_waypoints) >= 8:
            self.logger.record("waypoints/mean_per_episode",
                sum(self.episode_waypoints) / len(self.episode_waypoints))
            self.logger.record("waypoints/max_per_episode",
                max(self.episode_waypoints))
            self.episode_waypoints = []

        return True

class WaypointRewardShaping(gym.Wrapper):
    def __init__(self, env, gamma=0.5):
        super().__init__(env)
        self.gamma = gamma
        self.previous_distance = 0.0
        self.previous_action = None

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

    # def step(self, action):
    #     obs, reward, terminated, truncated, info = self.env.step(action)
    #     if reward >= 10:
    #         # self.previous_distance = 0.0
    #         self.previous_distance = self._get_distance(obs)

    #     current_distance = self._get_distance(obs)

    #     # Progress shapping
    #     shaping = -0.01  # Time step penalty
    #     if self.previous_distance != 0.0:
    #         shaping += self.gamma * (self.previous_distance - current_distance)
        
    #     # Smoothness 
    #     # if (self.previous_action is not None):
    #     #     action_diff = np.linalg.norm(action - self.previous_action)
    #     #     shaping -= 0.05 * action_diff
    #     self.previous_distance = current_distance
    #     self.previous_action = action
    #     return obs, reward + shaping, terminated, truncated, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_distance = self._get_distance(obs)

        shaping = -0.005
        # Skip shaping on the waypoint-capture step to avoid the spike
        if self.previous_distance != 0.0 and reward < 10:
            shaping += self.gamma * (self.previous_distance - current_distance)

        self.previous_distance = current_distance
        self.previous_action = action
        return obs, reward + shaping, terminated, truncated, info

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

def ppo(flight_mode, run, flight_dome_size, num_targets):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = flight_mode
    env_kwargs["flight_dome_size"] = flight_dome_size
    env_kwargs["num_targets"] = num_targets


    # 2. Create the vectorized environment using the custom builder
    # We create a list of 8 independent environments using a list comprehension
    env = SubprocVecEnv([
        make_custom_env("PyFlyt/QuadX-Waypoints-v4", env_kwargs, i) 
        for i in range(8)
    ])

    # 3. Apply the standard SB3 vector wrappers
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # env = VecNormalize.load("models/waypoint/test4-mode6-ppo_vecnormalize.pkl", env)
    # env.training = True   # re-enable stat updates for the new stage
    # env.norm_reward = True

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.005,       # slight reduction, let policy sharpen
        gae_lambda=0.95,      # was 0.9, better credit assignment over long episodes
        clip_range=0.2,
        n_epochs=10,          # add this — default is 10 but make it explicit
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        device=device,
    )
    # model = PPO.load("models/waypoint/test4-mode6-ppo", env=env)
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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)

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
    parser.add_argument("--steps", type=int, default=5000000) # Increased default steps for navigation
    parser.add_argument("--name", type=str, required=True) 
    parser.add_argument("--flight_dome_size", type=float, default=150.0) 
    parser.add_argument("--num_targets", type=int, default=4) 
    args = parser.parse_args()
    args.algo = args.algo.lower()

    NAME = f"{args.name}-mode{args.flight_mode}-{args.algo}"

    run = wandb.init(
        entity="ChelseaCity",
        project="RL-Drone-Project",
        name=NAME,
        config={
            "environment": "QuadX-Waypoints-v4",
            "algorithm": args.algo,
            "flight_mode": args.flight_mode,
            "total_timesteps": args.steps,
            "flight_dome_size": args.flight_dome_size,
            "num_targets": args.num_targets,
        },
        sync_tensorboard=True, 
        save_code=True,
    )

    if args.algo == "ppo":
        model, env = ppo(args.flight_mode, run, args.flight_dome_size, args.num_targets)
    elif args.algo == "sac":
        model, env = sac(args.flight_mode, run)
    else: 
        raise ValueError("Unknown algo!")


    print(f"Training Waypoints started on Flight Mode {args.flight_mode} with {args.algo.upper()}...", flush=True)
    print(f"Env: flight_dome_size:{args.flight_dome_size}, num_targets:{args.num_targets}", flush=True)
    
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
            WaypointMetricsCallback(),
            WandbCallback(
                verbose=1,
            ),
        ]),
        reset_num_timesteps=False
    )
        
    model.save(f"models/waypoint/{NAME}")
    env.save(f"models/waypoint/{NAME}_vecnormalize.pkl")
    run.finish()