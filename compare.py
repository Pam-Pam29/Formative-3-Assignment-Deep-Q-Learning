"""
================================================
  compare.py — MLP vs CNN Policy Comparison
  DQN on ALE/Breakout-v5
  100K steps each — quick comparison
  Stable Baselines 3 + Gymnasium
================================================
"""

import os
import csv
import numpy as np
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

gym.register_envs(ale_py)

TOTAL_TIMESTEPS = 100_000
SEED            = 42
ENV_ID          = "ALE/Breakout-v5"

SHARED_PARAMS = dict(
    learning_rate           = 0.0001,
    gamma                   = 0.98,
    batch_size              = 64,
    buffer_size             = 100_000,
    learning_starts         = 10_000,
    train_freq              = 4,
    target_update_interval  = 1_000,
    exploration_initial_eps = 1.0,
    exploration_final_eps   = 0.05,
    exploration_fraction    = 0.10,
    optimize_memory_usage   = False,
    verbose                 = 0,
)

for d in ["./models", "./eval_logs", "./logs"]:
    os.makedirs(d, exist_ok=True)


def make_env(seed):
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


def train_policy(policy_name, policy_str):
    print("\n" + "=" * 60)
    print(f"  Training : {policy_name}  |  Steps: {TOTAL_TIMESTEPS:,}")
    print("=" * 60)

    tag          = policy_name.lower().replace("policy", "")
    exp_eval_dir = f"./eval_logs/compare_{tag}"
    os.makedirs(exp_eval_dir, exist_ok=True)

    train_env = make_env(SEED)
    eval_env  = make_env(SEED + 1)

    model = DQN(policy=policy_str, env=train_env, seed=SEED, **SHARED_PARAMS)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = exp_eval_dir,
        log_path             = exp_eval_dir,
        eval_freq            = 25_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = eval_cb,
        reset_num_timesteps = True,
        progress_bar        = True,
    )

    model.save(f"./models/dqn_compare_{tag}")

    eval_npz    = f"{exp_eval_dir}/evaluations.npz"
    mean_reward = 0.0
    std_reward  = 0.0
    noted       = "No eval data."
    exp_csv     = f"./logs/compare_{tag}_reward_log.csv"

    if os.path.exists(eval_npz):
        data        = np.load(eval_npz)
        timesteps   = data["timesteps"]
        all_results = data["results"]
        all_lengths = data["ep_lengths"]
        final       = all_results[-1]
        mean_reward = float(np.mean(final))
        std_reward  = float(np.std(final))
        trend       = all_results.mean(axis=1)
        noted = (
            "Reward improving." if trend[-1] > trend[0]
            else "Reward declined." if trend[-1] < trend[0]
            else "Reward flat."
        )
        with open(exp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "mean_reward", "std_reward", "mean_ep_length"])
            for t, r, l in zip(timesteps, all_results, all_lengths):
                writer.writerow([int(t), round(float(np.mean(r)), 2),
                                 round(float(np.std(r)), 2),
                                 round(float(np.mean(l)), 1)])
        print(f"  Log → {exp_csv}")

    train_env.close()
    eval_env.close()

    return {
        "policy"      : policy_name,
        "mean_reward" : round(mean_reward, 2),
        "std_reward"  : round(std_reward,  2),
        "noted"       : noted,
    }


if __name__ == "__main__":

    print("=" * 60)
    print("  MLP vs CNN — DQN Breakout  (100K steps each ~20 min)")
    print("  Author: [YOUR NAME HERE]")
    print("=" * 60)

    cnn = train_policy("CnnPolicy", "CnnPolicy")
    mlp = train_policy("MlpPolicy", "MlpPolicy")

    results = [cnn, mlp]

    print("\n" + "=" * 60)
    print("  MLP vs CNN — Results")
    print("=" * 60)
    print(f"  {'Policy':<12} {'Mean':>8} {'Std':>8}  Note")
    print(f"  {'-'*12} {'-'*8} {'-'*8}  {'-'*25}")
    for r in results:
        print(f"  {r['policy']:<12} {r['mean_reward']:>8.2f} {r['std_reward']:>8.2f}  {r['noted']}")

    winner = max(results, key=lambda x: x["mean_reward"])
    loser  = min(results, key=lambda x: x["mean_reward"])
    print(f"\n  Winner     : {winner['policy']}  ({winner['mean_reward']:.2f})")
    print(f"  Weaker     : {loser['policy']}   ({loser['mean_reward']:.2f})")
    print(f"  Difference : +{(winner['mean_reward'] - loser['mean_reward']):.2f} "
          f"in favour of {winner['policy']}")

    print("""
  Why CNNPolicy wins on Atari:
  CNNPolicy uses convolutional layers to extract spatial features
  from 84x84 pixel frames — edges, objects, motion — before making
  Q-value decisions. MlpPolicy flattens all pixels into a 1D vector,
  destroying spatial structure entirely. CNN is the standard
  architecture for pixel-based RL (DeepMind DQN, 2013).
    """)

    csv_path = "./logs/mlp_vs_cnn_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["policy","mean_reward","std_reward","noted"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV → {csv_path}")
    print("\n  Done! Paste results into README.")
