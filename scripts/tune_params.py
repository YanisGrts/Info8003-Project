import optuna
import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

# Import custom configurations and wrappers for Waypoints
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv

from train_waypoint import WaypointRewardShaping, make_custom_env


def optimize_ppo(trial):
    """
    Optuna objective function for tuning PPO on the Waypoints task.
    """
    # 1. Define hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.05, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    
    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    # 2. Set up the environment matching the training script exactly
    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = 6  # Using mode 6 as the default baseline
    
    # We use fewer parallel environments for tuning to balance speed and memory
    num_envs = 4 
    
    env = SubprocVecEnv([
        make_custom_env("PyFlyt/QuadX-Waypoints-v4", env_kwargs, i) 
        for i in range(num_envs)
    ])
    
    # Apply standard vector wrappers
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 3. Initialize the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        batch_size=batch_size,
        n_steps=n_steps,
        policy_kwargs=dict(net_arch=[256, 256]), # Matching train script's architecture
        verbose=0, 
        device="cpu"
    )

    # 4. Train the model for enough steps to establish a performance trajectory
    try:
        model.learn(total_timesteps=100_000)
    except Exception as e:
        env.close()
        raise optuna.exceptions.TrialPruned()

    # 5. Evaluate how well this combination did
    # IMPORTANT: Since we used VecNormalize, we MUST freeze normalization during eval.
    # Otherwise, it evaluates scaled rewards which ruins the Optuna scoring.
    env.training = False
    env.norm_reward = False
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    
    env.close()
    
    return mean_reward

def main():
    print("Starting Hyperparameter Tuning for QuadX Waypoints...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=30)

    print("\n" + "="*50)
    print("TUNING FINISHED!")
    print(f"Best trial reward: {study.best_value:.2f}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("="*50)

if __name__ == "__main__":
    main()