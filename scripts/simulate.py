import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import pybullet as p
import numpy as np

from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv, WaypointRewardShaping, ActionRepeat


def create_marker(client_id):
    sphere_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.5,
        rgbaColor=[1, 0, 0, 0.8],
        physicsClientId=client_id
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=sphere_visual,
        basePosition=[0, 0, 0],
        physicsClientId=client_id
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Drone-Project-Simulator")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, required=True)
    parser.add_argument("--flight_mode", type=int, default=6, choices=[-1, 0, 4, 6, 7])
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--flight_dome_size", type=float, default=150.0) 
    parser.add_argument("--num_targets", type=int, default=4) 
    parser.add_argument("--goal_reach_distance", type=float, default=4.0) 
    args = parser.parse_args()

    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"] = args.flight_mode
    env_kwargs["flight_dome_size"] = args.flight_dome_size
    env_kwargs["num_targets"] = args.num_targets
    env_kwargs["goal_reach_distance"] = args.goal_reach_distance
    env_kwargs["render_mode"] = "human"


    def make_env():
        env = gym.make("PyFlyt/QuadX-Waypoints-v4", **env_kwargs)
        env = WaypointRewardShaping(env, shaping_coef=0.01)
        env = FlattenWaypointEnv(env, max_waypoints=args.num_targets)
        env = ActionRepeat(env, n=4)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(args.vecnormalize, env)
    env.training = False
    env.norm_reward = False

    # Load the correct algorithm based on user input
    if args.algo.lower() == "ppo":
        model = PPO.load(args.model, env=env)
    elif args.algo.lower() == "sac":
        model = SAC.load(args.model, env=env)

    obs = env.reset()

    # New
    aviary = env.envs[0].unwrapped
    client_id = aviary._client

    # Clean up UI
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id)
    
    # Disable mouse picking so clicking and dragging doesn't apply forces to the drone
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=client_id)

    # Set initial camera ONCE — never touch it again in the loop
    p.resetDebugVisualizerCamera(
        cameraDistance=15.0,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=client_id
    )

    marker_body = create_marker(client_id)

    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Just update the red sphere — nothing else
        drone_pos, _ = p.getBasePositionAndOrientation(
            aviary.drones[0].Id, physicsClientId=client_id
        )
        p.resetBasePositionAndOrientation(
            marker_body, drone_pos, [0, 0, 0, 1],
            physicsClientId=client_id
        )

        if done[0]:
            obs = env.reset()
            aviary = env.envs[0].unwrapped
            client_id = aviary._client
            marker_body = create_marker(client_id)

    env.close()