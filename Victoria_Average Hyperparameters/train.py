"""
================================================
  train.py — DQN Breakout
  10 Experiments + Final Optimised Run
  Stable Baselines 3 + Gymnasium
================================================
"""

import os
import csv
import shutil
import numpy as np
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

gym.register_envs(ale_py)

ENV_ID          = "ALE/Breakout-v5"
SEED            = 42
TOTAL_TIMESTEPS = 500_000

for d in ["./models", "./checkpoints", "./eval_logs", "./logs"]:
    os.makedirs(d, exist_ok=True)

EXPERIMENTS = [
    dict(exp=1,  lr=0.0001, gamma=0.97, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    dict(exp=2,  lr=0.0001, gamma=0.98, batch=64, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    dict(exp=3,  lr=0.0001, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.05, eps_frac=0.12),
    dict(exp=4,  lr=0.0002, gamma=0.97, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    dict(exp=5,  lr=0.0002, gamma=0.98, batch=64, eps_start=1.0, eps_end=0.05, eps_frac=0.12),
    dict(exp=6,  lr=0.0002, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.05, eps_frac=0.15),
    dict(exp=7,  lr=0.0003, gamma=0.97, batch=32, eps_start=1.0, eps_end=0.05, eps_frac=0.10),
    dict(exp=8,  lr=0.0003, gamma=0.98, batch=64, eps_start=1.0, eps_end=0.05, eps_frac=0.12),
    dict(exp=9,  lr=0.0003, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.02, eps_frac=0.12),
    dict(exp=10, lr=0.0002, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.02, eps_frac=0.15),
]

BEST_CONFIG = dict(
    exp=11, lr=0.0001, gamma=0.98, batch=64,
    eps_start=1.0, eps_end=0.01, eps_frac=0.15,
)


def make_env(seed):
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


def run_experiment(cfg, tag=None, buffer_size=100_000, optimize_memory=False):
    exp_id       = cfg["exp"]
    tag          = tag or f"exp{exp_id:02d}"
    exp_eval_dir = f"./eval_logs/{tag}"
    exp_ckpt_dir = f"./checkpoints/{tag}"
    os.makedirs(exp_eval_dir, exist_ok=True)
    os.makedirs(exp_ckpt_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  Experiment {exp_id}  [{tag}]")
    print(f"  lr={cfg['lr']}  gamma={cfg['gamma']}  batch={cfg['batch']}")
    print(f"  eps_start={cfg['eps_start']}  eps_end={cfg['eps_end']}  eps_frac={cfg['eps_frac']}")
    print("=" * 60)

    train_env = make_env(SEED)
    eval_env  = make_env(SEED + 1)

    model = DQN(
        policy                  = "CnnPolicy",
        env                     = train_env,
        learning_rate           = cfg["lr"],
        gamma                   = cfg["gamma"],
        batch_size              = cfg["batch"],
        buffer_size             = buffer_size,
        learning_starts         = 20_000,
        train_freq              = 4,
        target_update_interval  = 1_000,
        exploration_initial_eps = cfg["eps_start"],
        exploration_final_eps   = cfg["eps_end"],
        exploration_fraction    = cfg["eps_frac"],
        optimize_memory_usage   = optimize_memory,
        verbose                 = 0,
        seed                    = SEED,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=250_000, save_path=exp_ckpt_dir,
        name_prefix=f"dqn_{tag}", verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = exp_eval_dir,
        log_path             = exp_eval_dir,
        eval_freq            = 50_000,
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 0,
    )

    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = CallbackList([checkpoint_cb, eval_cb]),
        reset_num_timesteps = True,
        progress_bar        = True,
    )

    model_path = f"./models/dqn_{tag}"
    model.save(model_path)

    eval_npz    = f"{exp_eval_dir}/evaluations.npz"
    mean_reward = 0.0
    std_reward  = 0.0
    noted       = "No eval data."
    exp_csv     = f"./logs/{tag}_reward_log.csv"

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
            "Reward improving across training." if trend[-1] > trend[0]
            else "Reward declined — possible instability." if trend[-1] < trend[0]
            else "Reward flat — may need more timesteps."
        )
        with open(exp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "mean_reward", "std_reward", "mean_ep_length"])
            for t, r, l in zip(timesteps, all_results, all_lengths):
                writer.writerow([int(t), round(float(np.mean(r)), 2),
                                 round(float(np.std(r)),  2),
                                 round(float(np.mean(l)), 1)])

    print(f"\n  Exp {exp_id} done — Mean Reward: {mean_reward:.2f}  Std: {std_reward:.2f}")
    print(f"  Note: {noted}")
    print(f"  Log saved → {exp_csv}")

    train_env.close()
    eval_env.close()

    return {
        "exp": exp_id, "tag": tag,
        "lr": cfg["lr"], "gamma": cfg["gamma"], "batch": cfg["batch"],
        "eps_start": cfg["eps_start"], "eps_end": cfg["eps_end"], "eps_frac": cfg["eps_frac"],
        "mean_reward": round(mean_reward, 2),
        "std_reward":  round(std_reward,  2),
        "noted": noted,
    }


# ── Run 10 experiments ────────────────────────────────────────────────────────
print("=" * 60)
print("  DQN Breakout — 10-Experiment Hyperparameter Sweep")
print(f"  Environment : {ENV_ID}")
print(f"  Timesteps   : {TOTAL_TIMESTEPS:,} per experiment")
print("=" * 60)

summary_rows = []
for cfg in EXPERIMENTS:
    row = run_experiment(cfg)
    summary_rows.append(row)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  All 10 Experiments Complete")
print("=" * 60)
print(f"  {'Exp':>4} {'lr':>8} {'gamma':>6} {'batch':>6} {'eps_end':>8} {'eps_frac':>9} {'MeanRew':>9} {'Std':>6}")
print(f"  {'-'*4} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*6}")
for row in summary_rows:
    print(f"  {row['exp']:>4} {row['lr']:>8} {row['gamma']:>6} {row['batch']:>6} "
          f"{row['eps_end']:>8} {row['eps_frac']:>9} {row['mean_reward']:>9.2f} {row['std_reward']:>6.2f}")

# Save summary CSV
summary_csv = "./logs/experiment_summary.csv"
fieldnames  = ["exp", "lr", "gamma", "batch", "eps_start", "eps_end", "eps_frac",
               "mean_reward", "std_reward", "noted"]
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow({k: row[k] for k in fieldnames})
print(f"\n  Summary CSV → {summary_csv}")

# Best of 10
best10 = max(summary_rows, key=lambda x: x["mean_reward"])
print(f"\n  Best of 10: Exp {best10['exp']:02d} — {best10['mean_reward']:.2f}")

# ── Run optimised experiment ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Running Final Optimised Experiment [expBEST]")
print("  Hyperparameters derived from sweep results")
print("=" * 60)

best_row = run_experiment(
    BEST_CONFIG,
    tag           = "expBEST",
    buffer_size   = 50_000,
    optimize_memory = True,
)
summary_rows.append(best_row)

# ── Select and save overall best model ───────────────────────────────────────
overall_best = max(summary_rows, key=lambda x: x["mean_reward"])
best_tag     = overall_best["tag"]
best_src     = f"./models/dqn_{best_tag}.zip"
best_dst     = "./models/dqn_model.zip"

if os.path.exists(best_src):
    shutil.copy(best_src, best_dst)

print("\n" + "=" * 60)
print("  All Experiments Complete")
print("=" * 60)
print(f"  Best model  : {best_tag} — {overall_best['mean_reward']:.2f}")
print(f"  Saved as    : {best_dst}")
print("\n  Run play.py next.")
