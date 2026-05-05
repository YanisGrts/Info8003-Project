import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dogfight_wrapper import DogfightSelfPlayEnv


def make_env(rank: int, opponent_policy=None, seed: int = 0):
    """Returns a callable that creates a single DogfightSelfPlayEnv."""
    def _init():
        env = DogfightSelfPlayEnv(
            team_size=1,
            opponent_policy=opponent_policy, 
            flatten_observation=True,
            render_mode=None,
            max_duration_seconds=60,
            agent_hz=30,
        )
        env = DogfightRewardShaping(env)
        # Seed the action space for reproducibility
        env.action_space.seed(seed + rank)
        return env
    return _init


def build_vec_env(n_envs: int, opponent_policy=None, seed: int = 0):
    """Build a monitored DummyVecEnv with n_envs parallel dogfight envs."""
    # We use DummyVecEnv (single process)
    # because PyBullet's shared memory does work with fork().
    env = DummyVecEnv([make_env(i, opponent_policy, seed) for i in range(n_envs)])
    env = VecMonitor(env)
    return env

class SelfPlayCallback(BaseCallback):
    """
    Every `snapshot_every` timesteps:
      1. Saves a frozen copy of the current policy.
      2. Loads it back and sets it as the opponent in all envs.

    This is the core of the self-play curriculum: the agent always
    faces a slightly weaker version of itself, giving a smooth
    difficulty gradient.
    """

    def __init__(
        self,
        snapshot_every: int,
        snapshot_dir: str,
        run_name: str,
        algo: str, 
        verbose: int = 1, 
    ):
        super().__init__(verbose)
        self.snapshot_every = snapshot_every
        self.snapshot_dir = snapshot_dir
        self.run_name = run_name
        self._last_snapshot_step = 0
        self._snapshot_idx = 0
        self.algo = algo
        os.makedirs(snapshot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        steps_since_snapshot = self.num_timesteps - self._last_snapshot_step

        if steps_since_snapshot >= self.snapshot_every:
            self._update_opponent()
            self._last_snapshot_step = self.num_timesteps

        return True  # returning False would stop training

    def _update_opponent(self):
        path = os.path.join(
            self.snapshot_dir,
            f"{self.run_name}_snapshot_{self._snapshot_idx}.zip",
        )

        self.model.save(path)

        # Load policy as a frozen opponent
        if self.algo == "sac":
            print("loading sac opponents")
            opponent = SAC.load(path)
        else:
            opponent = PPO.load(path)


        for env in self.training_env.envs:
            env.set_opponent_policy(opponent)

        self._snapshot_idx += 1

        if self.verbose >= 1:
            print(
                f"\n[SelfPlay] Step {self.num_timesteps:,} — "
                f"opponent updated to snapshot {self._snapshot_idx - 1} "
                f"(saved at {path})"
            )




class DogfightRewardShaping(gym.Wrapper):

    DAMAGE_REWARD     =  500.0 
    OWN_CRASH_PENALTY = -300.0  
    LOSS_PENALTY      = -150.0  

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    def set_opponent_policy(self, policy):
        """Forward to the underlying DogfightSelfPlayEnv."""
        self.env.set_opponent_policy(policy)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaping = 0.0   
        # Pour pouvoir acces à ceci j'ai modifié ce qui était renvoyé dans le info dans le wrapper. 
        # Je pense pas que ce soit faux vu que c est utilisé dans la reward 
        # MAis pas dans la policy directement. 
        damage = info.get("opponent_damage_dealt", 0.0)
        if damage > 0.0:
            shaping += self.DAMAGE_REWARD * damage

        if terminated:
            we_crashed = info.get("collision", False) or info.get("out_of_bounds", False)
            we_won     = info.get("team_win", False)

            if we_crashed and not we_won:
                shaping += self.OWN_CRASH_PENALTY

            if not we_won and not we_crashed:
                shaping += self.LOSS_PENALTY

        return obs, reward + shaping, terminated, truncated, info


def build_model_ppo(env, run, args) -> PPO:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_dogfight] Using device: {device}")

    if args.load_model:
        print(f"[train_dogfight] Resuming from {args.load_model}")
        model = PPO.load(args.load_model, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=f"runs/{run.id}",
            # ---- Core hyperparameters ----
            learning_rate=3e-4,
            n_steps=1024,          
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,

            ent_coef=0.02,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            device=device,
        )

    return model

def build_model_sac(env, run, args) -> SAC:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_dogfight] Using device: {device}")

    if args.load_model:
        print(f"[train_dogfight] Resuming from {args.load_model}")
        model = SAC.load(args.load_model, env=env, device=device)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=f"runs/{run.id}",
            device=device,
            learning_rate=0.0004,
            buffer_size=1_000_000,  
            learning_starts=2000,  
            batch_size=128,
            tau=0.01,
            gamma=0.98,
            train_freq=1,
            gradient_steps=-1,
            ent_coef="auto",
            target_entropy=-10
        )

    return model


def main():
    parser = argparse.ArgumentParser(description="Dogfight self-play training")
    parser.add_argument("--steps",          type=int,   default=2_000_000,
                        help="Total environment timesteps")
    parser.add_argument("--n_envs",         type=int,   default=8,
                        help="Number of parallel environments")
    parser.add_argument("--snapshot_every", type=int,   default=100_000,
                        help="Timesteps between self-play opponent snapshots")
    parser.add_argument("--phase",          type=int,   default=1,
                        help="Phase number (for naming the run)")
    parser.add_argument("--load_model",     type=str,   default=None,
                        help="Path to a .zip model to resume from")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--algo", type=str, default="ppo")
    args = parser.parse_args()

    NAME = f"dogfight-phase{args.phase}-{args.algo}"
    SNAPSHOT_DIR = f"models/dogfight/{NAME}_snapshots"
    os.makedirs("models/dogfight", exist_ok=True)

    initial_opponent = None
    if args.load_model and args.phase > 1:
        # When resuming, load the previous checkpoint as starting opponent
        print(f"[train_dogfight] Loading initial opponent from {args.load_model}")
        if args.algo =="sac":
            initial_opponent = SAC.load(args.load_model)
        else:
            initial_opponent = PPO.load(args.load_model)

    # Build env
    env = build_vec_env(args.n_envs, opponent_policy=initial_opponent, seed=args.seed)

    # W&B run
    run = wandb.init(
        entity="ChelseaCity",
        project="RL-Drone-Project",
        name=NAME,
        config={
            "algorithm": args.algo,
            "environment": "MAFixedwingDogfightEnvV2",
            "total_timesteps": args.steps,
            "n_envs": args.n_envs,
            "snapshot_every": args.snapshot_every,
            "phase": args.phase,
            "seed": args.seed,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # Build model
    if args.algo == "sac":
        model = build_model_sac(env, run, args)
    else:
        model = build_model_ppo(env, run, args)
    print(f"[train_dogfight] Model has {sum(p.numel() for p in model.policy.parameters()):,} parameters")

    selfplay_cb = SelfPlayCallback(
        snapshot_every=args.snapshot_every,
        snapshot_dir=SNAPSHOT_DIR,
        run_name=NAME,
        verbose=1,
        algo = args.algo
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path=f"models/dogfight/{NAME}_checkpoints/",
        name_prefix=NAME,
        verbose=1,
    )

    wandb_cb = WandbCallback(verbose=1)

    print(f"\n[train_dogfight] Starting training: {args.steps:,} steps, "
          f"{args.n_envs} envs, snapshot every {args.snapshot_every:,} steps\n")

    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([selfplay_cb, checkpoint_cb, wandb_cb]),
        reset_num_timesteps=(args.load_model is None),
    )


    final_path = f"models/dogfight/{NAME}"
    model.save(final_path)
    print(f"\n[train_dogfight] Final model saved to {final_path}.zip")

    run.finish()


if __name__ == "__main__":
    main()
