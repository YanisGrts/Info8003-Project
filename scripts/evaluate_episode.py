import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add this line here!
import PyFlyt.gym_envs 

# Import custom configurations
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv
from stable_baselines3 import PPO, SAC

def analyze_episode(model_path, algo="ppo", flight_mode=6):
    """Runs a single episode and plots step-by-step metrics."""
    
    # 1. Setup Environment
    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = flight_mode
    
    env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", **env_kwargs)
    # We need to know the attitude dimension to extract the target later
    attitude_dim = env.observation_space["attitude"].shape[0] 
    env = FlattenWaypointEnv(env, max_waypoints=4)
    
    # 2. Load Model
    if algo == "ppo":
        model = PPO.load(model_path)
    else:
        model = SAC.load(model_path)

    # 3. Initialize Tracking Lists
    obs, info = env.reset(seed=42)
    
    rewards = []
    distances = []
    actions_log = []
    
    # 4. Run exactly one episode
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Extract Data ---
        rewards.append(reward)
        actions_log.append(action)
        
        # In FlattenWaypointEnv, the targets are appended AFTER the attitude data
        # The closest target's X, Y, Z coordinates are the first 3 values after the attitude
        closest_target_vector = obs[attitude_dim : attitude_dim + 3]
        distance_to_target = np.linalg.norm(closest_target_vector)
        distances.append(distance_to_target)
        
        done = terminated or truncated

    env.close()

    # 5. Plot the Data
    steps = np.arange(len(rewards))
    actions_log = np.array(actions_log)
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Reward Evolution
    plt.subplot(3, 1, 1)
    plt.plot(steps, rewards, label="Step Reward", color='blue')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Reward Evolution over Time")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # Plot 2: Distance to Current Target
    plt.subplot(3, 1, 2)
    plt.plot(steps, distances, label="Distance to Target (m)", color='red')
    plt.title("Distance to Active Waypoint")
    plt.ylabel("Distance (meters)")
    plt.legend()
    plt.grid(True)

    # Plot 3: Actions Output (What the neural net is trying to do)
    plt.subplot(3, 1, 3)
    for i in range(actions_log.shape[1]):
        plt.plot(steps, actions_log[:, i], label=f"Action {i}")
    plt.title("Agent Actions Output [-1 to 1]")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Action Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("episode_analysis.png")
    plt.show()
    print("Analysis complete. Saved plot to episode_analysis.png")

if __name__ == "__main__":
    # Change these to match your saved model
    MODEL_PATH = "models/waypoints-mode6-ppo.zip"  
    ALGO = "ppo"
    FLIGHT_MODE = 6
    
    analyze_episode(MODEL_PATH, ALGO, FLIGHT_MODE)