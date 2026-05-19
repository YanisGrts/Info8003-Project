import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback, EvalCallback
import os
import argparse
import torch
from stable_baselines3.common.logger import configure

from env_config import get_env_kwargs
from wrappers.wrappers import FlattenWaypointEnv

def make_eval_env(env_id, env_kwargs, vec_normalize_path=None):
    """Single env for evaluation — no SubprocVecEnv needed."""
    env = DummyVecEnv([make_custom_env(env_id, env_kwargs, rank=99, render_mode="rgb_array")])
    env = VecMonitor(env)
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False   # freeze stats during eval
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    return env

class VideoLoggerCallback(BaseCallback):
    """Records a short eval episode and uploads it to WandB."""
    def __init__(self, eval_env, log_freq=50_000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            frames = []
            obs = self.eval_env.reset()
            for _ in range(300):  # ~10s at 30fps
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = self.eval_env.step(action)
                frame = self.eval_env.render()
                if frame is not None:
                    frame = frame[..., :3]  # drop the alpha channel: (H, W, 4) → (H, W, 3)
                    frames.append(frame)
                if done[0]:
                    break

            if frames:
                # WandB expects (time, H, W, C) → transpose to (time, C, H, W)
                video = np.stack(frames).transpose(0, 3, 1, 2)
                wandb.log({
                    "eval/video": wandb.Video(video, fps=30, format="mp4")
                })
        return True

class WaypointMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_waypoints = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            wp = info.get("num_targets_reached", info.get("final_info", {}).get("num_targets_reached", None))
            if wp is not None:
                self._episode_waypoints.append(wp)

        if len(self._episode_waypoints) >= 10:  # log every 10 completed episodes
            wandb.log({
                "waypoints/mean_reached": np.mean(self._episode_waypoints),
                "waypoints/max_reached": np.max(self._episode_waypoints),
            })
            self._episode_waypoints = []

        return True

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

        shaping = self.shaping_coef * (self.previous_distance - current_distance)
        self.previous_distance = current_distance

        time_penalty = -0.1 # PPO
        # time_penalty = -0.05 # SAC

        raw_yaw_rate = self.env.unwrapped.env.state(0)[0][2]
        yaw_penalty = -0.01 * (raw_yaw_rate ** 2) # PPO
        # yaw_penalty = -0.001 * (raw_yaw_rate ** 2) # SAC

        custom_reward = shaping + time_penalty

        return obs, custom_reward, terminated, truncated, info

def make_custom_env(env_id, env_kwargs, rank, seed=0, render_mode=None):
    """Utility function to chain multiple wrappers for a multiprocessed env."""
    def _init():
        # Base
        if render_mode is None:
            env = gym.make(env_id, **env_kwargs)
        else:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
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
        for i in range(args.n_envs)
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
        # env = VecNormalize(env, norm_obs=True, norm_reward=False)#True)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=f"runs/{run.id}",
            learning_rate=2e-4,
            n_steps=8192,
            batch_size=512,
            ent_coef=0.01,
            gae_lambda=0.95,
            clip_range=0.1,
            use_sde=False,
            vf_coef=0.5,
            policy_kwargs=dict(
                net_arch=[256, 256, 256],
            ),
            device=device,
        )
        # model = PPO(
        #     "MlpPolicy",
        #     env,
        #     verbose=0,
        #     tensorboard_log=f"runs/{run.id}",
        #     learning_rate=3e-4,     # Standard starting LR
        #     n_steps=4096,           # Increased horizon
        #     batch_size=256,
        #     ent_coef=0.01,
        #     gae_lambda=0.95,        # Restored to 0.95 for better long-term credit assignment
        #     clip_range=0.2,
        #     use_sde=True,           # ADDED SDE for smoother flight
        #     policy_kwargs=dict(
        #         net_arch=[256, 256], 
        #     ),
        #     device=device,
        # )
        
    print(f"Using device: {model.device}")
    return model, env

def sac(args, run):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = args.flight_mode
    env_kwargs["flight_dome_size"] = args.dome_size
    env_kwargs["num_targets"] = args.num_waypoints

    env = SubprocVecEnv([
        make_custom_env("PyFlyt/QuadX-Waypoints-v4", env_kwargs, i) 
        for i in range(args.n_envs)
    ])

    if args.load_model is not None:
        print(f"Loading previous model and normalization stats from: {args.load_model}")
        
        # Load the normalization stats 
        vec_norm_path = f"{args.load_model}_vecnormalize.pkl"
        env = VecMonitor(env)
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True

        # Load the PPO model
        custom_objects = {
            "learning_rate": 3e-5,   # Drop from the initial 3e-4 for fine-tuning
            "tau": 0.005,            # Soft update coefficient
            "target_entropy": -4.0,
        }
        model = SAC.load(args.load_model, env=env, device=device, custom_objects=custom_objects)
        
        # Set up the new logger for this specific Phase
        new_logger = configure(f"runs/{run.id}", ["csv", "tensorboard"])
        model.set_logger(new_logger)
    else:
        print("Initializing completely new SAC model...")
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=False)#True)
      
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=f"runs/{run.id}",
            learning_rate=3e-4,
            buffer_size=500_000,      # Replay buffer — SAC needs this, PPO doesn't
            batch_size=512,
            tau=0.005,                # Soft update rate for target networks
            gamma=0.99,
            train_freq=16,             # Update every step (off-policy)
            gradient_steps=8,
            ent_coef="auto",          # SAC auto-tunes entropy — leave this as "auto"
            target_entropy=-4.0,
            use_sde=True,             # Same SDE trick your colleague used for smoother flight
            sde_sample_freq=64,
            policy_kwargs=dict(
                net_arch=[256, 256],
                log_std_init=-3,
            ),
            learning_starts=5_000,
            device=device,
        )

    eval_env_kwargs = get_env_kwargs("waypoints")
    eval_env_kwargs["flight_mode"] = args.flight_mode
    eval_env_kwargs["flight_dome_size"] = args.dome_size
    eval_env_kwargs["num_targets"] = args.num_waypoints

    vec_norm_path = f"{args.load_model}_vecnormalize.pkl" if args.load_model else None
    eval_env = make_eval_env("PyFlyt/QuadX-Waypoints-v4", eval_env_kwargs, vec_normalize_path=None)

    print(f"Using device: {model.device}")
    return model, env, eval_env



if __name__ == "__main__":
    print(f"CPU cores visible: {os.cpu_count()}", flush=True)

    parser = argparse.ArgumentParser(description="RL-Drone-Project-Waypoints")
    # Note: For waypoints, flight mode 6 (velocity control) or 7 (position control) are usually easier to start with
    parser.add_argument("--flight_mode", type=int, default=6, choices=[-1,0,4,6,7])
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--steps", type=int, default=1000000) # Increased default steps for navigation

    parser.add_argument("--dome_size", type=float, default=20.0, help="Radius of the waypoint spawn dome")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a previously trained model (.zip)")
    parser.add_argument("--phase", type=int, default=1, help="Phase number for naming the run")
    parser.add_argument("--num_waypoints", type=int, default=1, help="Number of active targets")
    parser.add_argument("--run_id", type=str, default="", help="Run label e.g. RunA, RunB, RunC")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments.")

    args = parser.parse_args()
    args.algo = args.algo.lower()

    NAME = f"waypoints-mode{args.flight_mode}-{args.algo}-Phase{args.phase}-Dome{int(args.dome_size)}-Wp{args.num_waypoints}-{args.run_id}"

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
        model, env, eval_env = sac(args, run)
    else: 
        raise ValueError("Unknown algo!")

    print(f"Training Waypoints started on Flight Mode {args.flight_mode} with {args.algo.upper()}...")
    
    os.makedirs("models", exist_ok=True)
    
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=max(100_000 // 8, 1),  # every ~100k env steps, adjusted for num_envs
    #     save_path=f"models/waypoint_phase/{NAME}_checkpoints/",
    #     name_prefix=NAME,
    #     save_vecnormalize=True,
    # )

    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([
            WaypointMetricsCallback(),
            # VideoLoggerCallback(eval_env, log_freq=50_000),
            EvalCallback(
                eval_env,
                best_model_save_path=f"models/waypoint_phase/{NAME}_best/",
                log_path=f"runs/{run.id}",
                eval_freq=max(100_000 // args.n_envs, 1),  # every ~10k env steps
                n_eval_episodes=5,
                deterministic=True,
                render=False,
            ),
            WandbCallback(verbose=1),
        ]),
    )


    model.save(f"models/waypoint_phase/{NAME}")
    env.save(f"models/waypoint_phase/{NAME}_vecnormalize.pkl")
    run.finish()