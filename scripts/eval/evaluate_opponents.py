"""
Evaluate a trained dogfight agent against a pool of handcrafted opponents.
Outcome logic matches the official tournament rules exactly:

  * Winner = higher accumulated PyFlyt reward at episode end.
  * Collision / out-of-bounds → −1000 terminal penalty applied before comparing.
  * Kill = opponent HP reaches 0 from combat damage (received_hits > 0 AND enemy dead).
  * Dmg% = average fraction of enemy HP removed per game.

Usage:
  python scripts/evaluate_opponents.py --model models/dogfight/my_model.zip
  python scripts/evaluate_opponents.py --model models/dogfight/my_model.zip --n_episodes 20
  python scripts/evaluate_opponents.py --model models/dogfight/my_model.zip \
      --opponents aggressive evasive circling_right
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dogfight.opponent_agents import ALL_OPPONENTS
from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
COLLISION_PENALTY = -1000.0   # must match the tournament runner


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str):
    if path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("submission", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_model()
    for cls in [PPO, SAC]:
        try:
            return cls.load(path)
        except Exception:
            continue
    raise ValueError(f"Could not load model from {path}")



def run_episode(env, model, opponent, seed: int = 0):
    """
    Run one episode on a persistent env instance.

    Args:
        env: Reused MAFixedwingDogfightEnvV2 — call env.reset() between episodes,
             do NOT close it here.

    Returns a dict with:
      outcome        "win" | "loss" | "draw"
      our_reward     our accumulated reward (after collision penalty)
      their_reward   opponent accumulated reward (after collision penalty)
      steps          episode length
      we_killed      True if opponent HP→0 from combat (not a crash)
      we_crashed     True if we collided or went out of bounds
      they_crashed   True if opponent collided or went out of bounds
      damage_dealt   fraction of enemy HP removed [0, 1]
    """
    if hasattr(opponent, "reset"):
        opponent.reset()

    observations, _ = env.reset(seed=seed)

    acc_rewards = {"uav_0": 0.0, "uav_1": 0.0}
    steps       = 0
    final_infos = {}

    while env.agents:
        actions = {}
        for agent in env.agents:
            obs = observations[agent]
            if agent == "uav_0":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = opponent.predict(obs, deterministic=True)
            actions[agent] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent, reward in rewards.items():
            acc_rewards[agent] = acc_rewards.get(agent, 0.0) + reward

        for agent, info in infos.items():
            if terminations.get(agent, False) or truncations.get(agent, False):
                final_infos[agent] = info

        steps += 1

    # ── Apply collision penalty (tournament rule) ─────────────────────────
    our_info   = final_infos.get("uav_0", {})
    their_info = final_infos.get("uav_1", {})

    we_crashed   = our_info.get("collision",   False) or our_info.get("out_of_bounds",   False)
    they_crashed = their_info.get("collision", False) or their_info.get("out_of_bounds", False)

    if we_crashed:
        acc_rewards["uav_0"] += COLLISION_PENALTY
    if they_crashed:
        acc_rewards["uav_1"] += COLLISION_PENALTY

    our_reward   = acc_rewards["uav_0"]
    their_reward = acc_rewards["uav_1"]

    # ── Win condition: higher reward wins ─────────────────────────────────
    if our_reward > their_reward:
        outcome = "win"
    elif their_reward > our_reward:
        outcome = "loss"
    else:
        outcome = "draw"

    # ── Kill: opponent HP→0 from actual combat (not self-crash) ──────────
    enemy_hp_zero = their_info.get("health", 1.0) == 0.0
    enemy_hit     = their_info.get("received_hits", 0) > 0
    we_killed     = enemy_hp_zero and enemy_hit and not they_crashed

    # ── Damage dealt (only from hits, not opponent's own crash) ───────────
    damage_dealt = max(0.0, 1.0 - their_info.get("health", 1.0))
    if they_crashed and not enemy_hit:
        damage_dealt = 0.0

    return {
        "outcome":      outcome,
        "our_reward":   our_reward,
        "their_reward": their_reward,
        "steps":        steps,
        "we_killed":    we_killed,
        "we_crashed":   we_crashed,
        "they_crashed": they_crashed,
        "damage_dealt": damage_dealt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate against one opponent type
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vs_opponent(model, opponent_name: str, opponent,
                          n_episodes: int, render: bool = False):
    results = []
    env = MAFixedwingDogfightEnvV2(
        team_size=1,
        assisted_flight=True,
        flatten_observation=True,
        render_mode="human" if render else None,
        max_duration_seconds=60,
        agent_hz=30,
    )

    try:
        for i in range(n_episodes):
            ep = run_episode(env, model, opponent, seed=100 + i)
            results.append(ep)

            symbol = {"win": "✓", "loss": "✗", "draw": "~"}[ep["outcome"]]
            tags = ""
            if ep["we_killed"]:    tags += " [KILL]"
            if ep["we_crashed"]:   tags += " [WE CRASHED]"
            if ep["they_crashed"]: tags += " [THEY CRASHED]"

            print(f"  [{opponent_name}] ep {i+1:>2}/{n_episodes}  "
                  f"{symbol}  "
                  f"our={ep['our_reward']:+8.1f}  "
                  f"their={ep['their_reward']:+8.1f}  "
                  f"steps={ep['steps']}"
                  f"{tags}")
    finally:
        env.close()

    outcomes    = [r["outcome"]      for r in results]
    our_rewards = [r["our_reward"]   for r in results]
    kills       = [r["we_killed"]    for r in results]
    damages     = [r["damage_dealt"] for r in results]
    crashes     = [r["we_crashed"]   for r in results]

    wins   = outcomes.count("win")
    losses = outcomes.count("loss")
    draws  = outcomes.count("draw")

    return {
        "opponent":    opponent_name,
        "wins":        wins,
        "losses":      losses,
        "draws":       draws,
        "win_rate":    wins         / n_episodes,
        "kill_rate":   sum(kills)   / n_episodes,
        "crash_rate":  sum(crashes) / n_episodes,
        "mean_damage": float(np.mean(damages)),
        "mean_reward": float(np.mean(our_rewards)),
        "std_reward":  float(np.std(our_rewards)),
        "mean_steps":  float(np.mean([r["steps"] for r in results])),
        "rewards":     our_rewards,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results: list, model_name: str, save_path: str = None):
    names       = [r["opponent"]    for r in all_results]
    win_rates   = [r["win_rate"]    for r in all_results]
    kill_rates  = [r["kill_rate"]   for r in all_results]
    crash_rates = [r["crash_rate"]  for r in all_results]
    damages     = [r["mean_damage"] for r in all_results]
    wins        = [r["wins"]        for r in all_results]
    losses      = [r["losses"]      for r in all_results]
    draws       = [r["draws"]       for r in all_results]
    n_ep        = wins[0] + losses[0] + draws[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Dogfight evaluation — {model_name}\n"
        f"Win = higher reward after collision penalty (−1000)",
        fontsize=12, fontweight="bold",
    )

    x     = np.arange(len(names))
    bar_w = 0.55

    # ── Win rate ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    colors = ["#2ecc71" if w >= 0.6 else "#e74c3c" if w < 0.4 else "#f39c12"
              for w in win_rates]
    bars = ax.bar(x, win_rates, width=bar_w, color=colors, edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="50%")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Win rate per opponent")
    ax.legend(fontsize=8)
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{wr:.0%}", ha="center", fontsize=8)

    # ── Kill rate vs crash rate ───────────────────────────────────────────
    ax = axes[0, 1]
    w2 = bar_w / 2
    ax.bar(x - w2 / 2, kill_rates,  width=w2, label="Kill rate (shot down)", color="#3498db")
    ax.bar(x + w2 / 2, crash_rates, width=w2, label="Our crash rate",        color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Kill rate vs our crash rate")
    ax.legend(fontsize=8)

    # ── Mean damage dealt ─────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(x, [d * 100 for d in damages], width=bar_w, color="#9b59b6", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean damage dealt (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Average damage dealt per game")
    for i, d in enumerate(damages):
        ax.text(i, d * 100 + 2, f"{d:.0%}", ha="center", fontsize=8)

    # ── Win / Draw / Loss stacked ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.bar(x, wins,   width=bar_w, label="Wins",   color="#2ecc71", edgecolor="white")
    ax.bar(x, draws,  width=bar_w, label="Draws",  color="#f39c12", edgecolor="white",
           bottom=wins)
    ax.bar(x, losses, width=bar_w, label="Losses", color="#e74c3c", edgecolor="white",
           bottom=[w + d for w, d in zip(wins, draws)])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"Episodes (out of {n_ep})")
    ax.set_title("Outcome breakdown")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list):
    print("\n" + "=" * 88)
    print(f"{'Opponent':<20} {'Win%':>6} {'Kill%':>6} {'Crash%':>7} "
          f"{'Dmg%':>6} {'W':>4} {'L':>4} {'D':>4}  {'Mean reward':>12}")
    print("-" * 88)
    for r in sorted(all_results, key=lambda x: -x["win_rate"]):
        print(f"  {r['opponent']:<18} "
              f"{r['win_rate']:>5.0%} "
              f"{r['kill_rate']:>6.0%} "
              f"{r['crash_rate']:>7.0%} "
              f"{r['mean_damage']:>5.0%} "
              f"{r['wins']:>4} {r['losses']:>4} {r['draws']:>4}  "
              f"{r['mean_reward']:>+12.1f}")
    print("=" * 88)

    total_w   = sum(r["wins"]        for r in all_results)
    total_l   = sum(r["losses"]      for r in all_results)
    total_d   = sum(r["draws"]       for r in all_results)
    total     = total_w + total_l + total_d
    avg_kill  = np.mean([r["kill_rate"]   for r in all_results])
    avg_crash = np.mean([r["crash_rate"]  for r in all_results])
    avg_dmg   = np.mean([r["mean_damage"] for r in all_results])
    print(f"\n  Overall  {total_w/total:.0%} win  |  "
          f"{avg_kill:.0%} kill  |  "
          f"{avg_crash:.0%} crash  |  "
          f"{avg_dmg:.0%} avg damage\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a dogfight agent vs handcrafted opponents (tournament rules)"
    )
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Games per opponent (default: 10)")
    parser.add_argument("--opponents",  type=str, nargs="+", default=None,
                        help=f"Subset to test. Available: {list(ALL_OPPONENTS.keys())}")
    parser.add_argument("--render",     action="store_true",
                        help="Render first episode of each matchup")
    parser.add_argument("--output",     type=str, default="dogfight_eval.png")
    args = parser.parse_args()

    if args.opponents:
        unknown = set(args.opponents) - set(ALL_OPPONENTS)
        if unknown:
            print(f"Unknown opponents: {unknown}")
            sys.exit(1)
        opponents = {k: ALL_OPPONENTS[k] for k in args.opponents}
    else:
        opponents = ALL_OPPONENTS

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model)
    model_name = os.path.splitext(os.path.basename(args.model))[0]

    print(f"Evaluating against {len(opponents)} opponent(s), "
          f"{args.n_episodes} episodes each.\n")
    print("Win rule: higher accumulated reward wins. Collision/OOB → −1000 penalty.\n")

    all_results = []
    for name, opponent in opponents.items():
        print(f"\n── vs {name} ──")
        result = evaluate_vs_opponent(
            model, name, opponent, args.n_episodes, render=args.render
        )
        all_results.append(result)

    print_summary(all_results)
    plot_results(all_results, model_name, save_path=args.output)


if __name__ == "__main__":
    main()
