import matplotlib.pyplot as plt
import numpy as np
import gymnasium
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv
import PyFlyt.gym_envs 
def analyze_episode(model_path, algo="ppo", flight_mode=6, norm_path=None):
    """Runs a single episode and plots step-by-step metrics."""
    
    # 1. Setup Environment
    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = flight_mode
    
    # Use a function for DummyVecEnv to ensure a clean instance
    def make_env():
        env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", **env_kwargs)
        env = FlattenWaypointEnv(env, max_waypoints=4)
        return env

    # We need the attitude dimension from a temporary env to know where to slice
    temp_env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", **env_kwargs)
    attitude_dim = temp_env.observation_space["attitude"].shape[0]
    temp_env.close()
    
    vec_env = DummyVecEnv([make_env])
    
    # Load Normalization
    try:
        env = VecNormalize.load(norm_path, vec_env)
    except Exception as e:
        print(f"Could not load normalization from {norm_path}: {e}")
        env = vec_env # Fallback to non-normalized if file is missing
        
    env.training = False      
    env.norm_reward = False   

    # 2. Load Model
    if algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        model = SAC.load(model_path, env=env)

    # 3. Initialize Tracking
    obs = env.reset()
    rewards = []
    distances = []
    actions_log = []
    
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action) 
        
        rewards.append(reward[0])
        actions_log.append(action[0])
        
        # --- FIX: UN-NORMALIZE THE OBSERVATION ---
        # If VecNormalize was used, obs are z-scores. 
        # Real = (Normalized * Std) + Mean
        if isinstance(env, VecNormalize):
            # obs[0] is the vector for the first env in the batch
            # env.obs_rms.var is the variance, sqrt is std
            real_obs = (obs[0] * np.sqrt(env.obs_rms.var)) + env.obs_rms.mean
        else:
            real_obs = obs[0]
            
        # Now slice the REAL observation to get meters
        closest_target_vector = real_obs[attitude_dim : attitude_dim + 3]
        distance_to_target = np.linalg.norm(closest_target_vector)
        distances.append(distance_to_target)
        
        
    env.close()

    # 5. Plotting
    steps = np.arange(len(rewards))
    actions_log = np.array(actions_log)
    
    plt.figure(figsize=(12, 12))
    
    # Plot 1: Reward
    plt.subplot(3, 1, 1)
    plt.plot(steps, rewards, label="Step Reward", color='blue')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Reward Evolution (Mode {flight_mode} {algo})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # Plot 2: Distance (Now in real meters!)
    plt.subplot(3, 1, 2)
    plt.plot(steps, distances, label="Distance to Target (m)", color='red')
    plt.title("Real Distance to Active Waypoint")
    plt.ylabel("Meters")
    plt.legend()
    plt.grid(True)

    # Plot 3: Actions
    plt.subplot(3, 1, 3)
    labels = ["Roll", "Pitch", "Yaw", "Throttle"] # Standard PyFlyt mapping
    for i in range(actions_log.shape[1]):
        plt.plot(steps, actions_log[:, i], label=labels[i] if i < len(labels) else f"Act {i}")
    plt.title("Neural Network Output (Action Space)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Value [-1, 1]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("episode_analysis.png")
    plt.show()
    print("Analysis complete. Saved plot to episode_analysis.png")

if __name__ == "__main__":
    import argparse 
    # Change these to match your saved model
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="RL algorithm used for training")
    parser.add_argument("--flight_mode", type=int, default=6, choices=[-1,0,4,6,7])
    parser.add_argument("--norm_path", type=str, default=None, help="Path to VecNormalize .pkl file")
    args = parser.parse_args()
    MODEL_PATH = args.model
    ALGO = args.algo
    FLIGHT_MODE = args.flight_mode
    NORM_PATH = args.norm_path
    analyze_episode(MODEL_PATH, ALGO, FLIGHT_MODE, NORM_PATH)