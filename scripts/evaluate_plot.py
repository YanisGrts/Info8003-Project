"""
Evaluation script for trained RL agents on PyFlyt environments.
Computes statistics, optionally renders episodes, and plots 3D trajectories.
"""

import argparse
import json
import os
import sys

import gymnasium
import numpy as np
import PyFlyt.gym_envs

# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import SB3 VecEnv wrappers to match training
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv


def make_env(env_id, flight_mode=0, render_mode=None, env_kwargs=None):
    """Create a PyFlyt environment."""
    env = gymnasium.make(env_id, flight_mode=flight_mode, render_mode=render_mode,
                         **(env_kwargs or {}))
    if isinstance(env.observation_space, gymnasium.spaces.Dict):
        env = FlattenWaypointEnv(env, max_waypoints=4)
    return env


def load_model(model_path):
    """Load a model from a .py submission module or .zip SB3 checkpoint."""
    if model_path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("submission", model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_model()
    # Legacy SB3 .zip fallback
    from stable_baselines3 import PPO, SAC
    for cls in [PPO, SAC]:
        try:
            return cls.load(model_path)
        except Exception:
            continue
    raise ValueError(f"Could not load model: {model_path}")


def get_drone_position(env, obs):
    """Extracts the drone's [x, y, z] position from the unnormalized observation array."""
    try:
        # If the environment is wrapped in VecNormalize, we must get the unscaled observation
        if hasattr(env, "get_original_obs"):
            raw_obs = env.get_original_obs()
        else:
            raw_obs = obs
        
        # In DummyVecEnv, the shape is (1, 33).
        # Based on the environment space, linear position is at indices 10, 11, 12
        return raw_obs[0, 10:13].copy()
    except Exception as e:
        print(f"Warning: Could not extract position ({e})")
        return np.array([0.0, 0.0, 0.0])
def get_waypoints(env):
    """Attempts to extract target waypoints for plotting."""
    try:
        unwrapped = env.envs[0].unwrapped
        # PyFlyt >= 0.3.0 stores them in the .waypoints object
        if hasattr(unwrapped, "waypoints") and hasattr(unwrapped.waypoints, "targets"):
            return unwrapped.waypoints.targets.copy()
        # Older versions fallback
        if hasattr(unwrapped, "targets"):
            return unwrapped.targets.copy()
    except Exception as e:
        print(f"Warning: Could not extract waypoints ({e})")
        
    return []


def plot_trajectory_3d(trajectory, waypoints=None, env_id="PyFlyt Env"):
    """Plots the 3D trajectory of the drone."""
    if not trajectory:
        print("No trajectory data to plot.")
        return

    trajectory = np.array(trajectory)
    xs, ys, zs = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the drone trajectory
    ax.plot(xs, ys, zs, label='Drone Trajectory', color='blue', linewidth=2)
    
    # Mark start and end points
    ax.scatter(xs[0], ys[0], zs[0], color='green', s=100, label='Start Position', marker='o')
    ax.scatter(xs[-1], ys[-1], zs[-1], color='black', s=100, label='End Position', marker='X')

    # Plot waypoints if available
    if waypoints is not None and len(waypoints) > 0:
        waypoints = np.array(waypoints)
        wx, wy, wz = waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]
        ax.scatter(wx, wy, wz, color='red', s=100, label='Targets/Waypoints', marker='*')

    ax.set_title(f"3D Flight Trajectory - {env_id}")
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    
    # Make axes limits equal to have a realistic aspect ratio
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model(model_path, env_id, n_episodes=20, flight_mode=0,
                   render=False, deterministic=True, env_kwargs=None, 
                   vecnorm_path=None, plot_3d=False):
    """Evaluate a trained model and return detailed statistics."""
    model = load_model(model_path)

    render_mode = "human" if render else None
    
    def _init():
        return make_env(env_id, flight_mode=flight_mode, render_mode=render_mode, env_kwargs=env_kwargs)

    env = DummyVecEnv([_init])

    if vecnorm_path:
        print(f"Loading VecNormalize stats from: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    episode_rewards, episode_lengths, episode_crashes = [], [], []
    episode_waypoints = []
    
    # Storage for 3D plotting
    first_episode_trajectory = []
    first_episode_targets = []

    for i in range(n_episodes):
        env.env_method("reset", seed=42 + 16 + i)
        obs = env.reset()
        
        # Get targets for the first episode if we want to plot them
        if i == 0 and plot_3d:
            first_episode_targets = get_waypoints(env)
            # UPDATE THIS LINE: Add 'obs' to the arguments
            first_episode_trajectory.append(get_drone_position(env, obs))
        
        total_reward, steps, crashed = 0.0, 0, False

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            
            if i == 0 and plot_3d:
                first_episode_trajectory.append(get_drone_position(env, obs))
            
            total_reward += rewards[0]
            steps += 1
            
            if render:
                env.render()

            if dones[0]:
                info = infos[0]
                if "terminal_info" in info:
                    info = info["terminal_info"]
                    
                crashed = rewards[0] <= -50

                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                episode_crashes.append(crashed)
                episode_waypoints.append(info.get("num_targets_reached", 0))

                print(f"  Episode {i+1}/{n_episodes}: reward={total_reward:.2f}, "
                      f"steps={steps}, crashed={crashed}", end="")
                if "Waypoints" in env_id:
                    print(f", waypoints={episode_waypoints[-1]}", end="")
                print()
                break

    env.close()

    # Generate the plot for the first episode if requested
    if plot_3d and first_episode_trajectory:
        print("\nRendering 3D Trajectory for Episode 1...")
        plot_trajectory_3d(first_episode_trajectory, first_episode_targets, env_id)

    results = {
        "model_path": model_path,
        "env_id": env_id,
        "n_episodes": n_episodes,
        "flight_mode": flight_mode,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "crash_rate": float(np.mean(episode_crashes)),
        "episode_rewards": [float(r) for r in episode_rewards],
    }
    if "Waypoints" in env_id:
        results["mean_waypoints"] = float(np.mean(episode_waypoints))

    return results


def print_results(results):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Evaluation: {os.path.basename(results['model_path'])}")
    print(f"Env: {results['env_id']}")
    print(f"{'='*60}")
    print(f"  Episodes:      {results['n_episodes']}")
    print(f"  Mean reward:   {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"  Median reward: {results['median_reward']:.2f}")
    print(f"  Min/Max:       {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Mean length:   {results['mean_length']:.1f}")
    print(f"  Crash rate:    {results['crash_rate']*100:.1f}%")
    if "mean_waypoints" in results:
        print(f"  Mean waypoints: {results['mean_waypoints']:.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.zip)")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to the saved VecNormalize stats (.pkl)")
    parser.add_argument("--env", type=str, required=True, choices=["hover", "waypoints"])
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--flight_mode", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--dome_size", type=float, default=150.0)
    parser.add_argument("--num_targets", type=int, default=4)
    
    # --- ADDED FLAG FOR 3D PLOTTING ---
    parser.add_argument("--plot_3d", action="store_true", help="Plot the 3D trajectory of the first episode")
    
    args = parser.parse_args()

    env_map = {
        "hover": "PyFlyt/QuadX-Hover-v4",
        "waypoints": "PyFlyt/QuadX-Waypoints-v4",
    }

    env_kwargs = get_env_kwargs(args.env)
    
    if args.env == "waypoints":
        env_kwargs["flight_dome_size"] = args.dome_size
        env_kwargs["num_targets"] = args.num_targets

    results = evaluate_model(
        args.model, env_map[args.env], args.n_episodes, args.flight_mode, args.render,
        env_kwargs=env_kwargs, vecnorm_path=args.vecnorm, plot_3d=args.plot_3d
    )
    print_results(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()