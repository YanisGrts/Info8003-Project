import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import pybullet as p
import numpy as np

from env_config import get_env_kwargs
from wrappers.wrappers import FlattenWaypointEnv
from waypoint.train_waypoint_phase import WaypointRewardShaping, make_custom_env

# ── Helpers ───────────────────────────────────────────────────────────────────

def create_marker(client_id):
    sphere_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=1.5,
        rgbaColor=[1, 0, 0, 0.8],
        physicsClientId=client_id
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=sphere_visual,
        basePosition=[0, 0, 0],
        physicsClientId=client_id
    )

def get_raw_aviary(vec_env):
    """Unwrap all layers to reach the raw PyFlyt QuadXWaypointsEnv."""
    env = vec_env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env

def get_client_id(aviary):
    """Find the PyBullet physics client ID, printing all attrs if not found."""
    for attr in dir(aviary):
        try:
            val = getattr(aviary, attr)
            if isinstance(val, int) and "client" in attr.lower():
                return val
        except Exception:
            continue
    # Broader search — any int attr that could be a pybullet client
    candidates = {a: getattr(aviary, a) for a in dir(aviary)
                  if not a.startswith("__")
                  and isinstance(getattr(aviary, a, None), int)}
    raise RuntimeError(
        f"Cannot find client ID.\nClass: {type(aviary)}\nInt attributes: {candidates}"
    )

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Drone-Project-Simulator")
    parser.add_argument("--model",            type=str,   required=True)
    parser.add_argument("--vecnormalize",     type=str,   required=True)
    parser.add_argument("--flight_mode",      type=int,   default=6,     choices=[-1, 0, 4, 6, 7])
    parser.add_argument("--algo",             type=str,   default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--flight_dome_size", type=float, default=20.0,
                        help="Must match the dome size used during training")
    parser.add_argument("--num_waypoints",    type=int,   default=1,
                        help="Number of active targets (matches --num_waypoints in training)")
    parser.add_argument("--goal_reach_distance", type=float, default=4.0)
    args = parser.parse_args()

    # ── Build env (must mirror make_custom_env in train_waypoint_phase.py) ────
    env_kwargs = get_env_kwargs("waypoints")
    env_kwargs["flight_mode"]          = args.flight_mode
    env_kwargs["flight_dome_size"]     = args.flight_dome_size
    env_kwargs["num_targets"]          = args.num_waypoints
    env_kwargs["goal_reach_distance"]  = args.goal_reach_distance
    env_kwargs["render_mode"]          = "human"


    env = DummyVecEnv([make_custom_env("PyFlyt/QuadX-Waypoints-v4", env_kwargs, rank=0)])
    env = VecNormalize.load(args.vecnormalize, env)
    env.training   = False   # freeze running stats
    env.norm_reward = False  # rewards not normalised at inference

    # ── Load model ────────────────────────────────────────────────────────────
    if args.algo.lower() == "ppo":
        model = PPO.load(args.model, env=env)
    elif args.algo.lower() == "sac":
        model = SAC.load(args.model, env=env)

    obs = env.reset()

    # ── PyBullet setup ────────────────────────────────────────────────────────
    aviary    = get_raw_aviary(env)
    client_id = get_client_id(aviary)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI,           0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,       0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=client_id)

    p.resetDebugVisualizerCamera(
        cameraDistance=args.flight_dome_size * 1.1,  # 1.8× the dome radius
        cameraYaw=20,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=client_id
    )

    marker_body = create_marker(client_id)

    # ── Simulation loop ───────────────────────────────────────────────────────
    for _ in range(10_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        drone_pos, _ = p.getBasePositionAndOrientation(
            aviary.drones[0].Id, physicsClientId=client_id
        )
        p.resetBasePositionAndOrientation(
            marker_body, drone_pos, [0, 0, 0, 1],
            physicsClientId=client_id
        )

        if done[0]:
            obs         = env.reset()
            aviary      = get_raw_aviary(env)
            client_id   = get_client_id(aviary)
            marker_body = create_marker(client_id)

    env.close()