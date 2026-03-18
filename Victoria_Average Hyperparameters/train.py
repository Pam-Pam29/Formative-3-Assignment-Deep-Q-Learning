"""
================================================
  train.py — DQN Breakout
  10 Experiments with Moderate Hyperparameters
  Stable Baselines 3 + Gymnasium
  All fixes applied:
    - ALE registration
    - VecTransposeImage on both envs (no type mismatch)
    - optimize_memory_usage=False (no buffer conflict)
    - Single np.load call (no double-load bug)
    - Immediate download link after each experiment
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
from IPython.display import FileLink, display

# ── Register ALE environments ─────────────────────────────────────────────────
gym.register_envs(ale_py)

# ── Output Dirs ───────────────────────────────────────────────────────────────
MODEL_DIR      = "./models"
CHECKPOINT_DIR = "./checkpoints"
EVAL_LOG_DIR   = "./eval_logs"
TB_LOG_DIR     = "./tb_logs"
LOG_DIR        = "./logs"

for d in [MODEL_DIR, CHECKPOINT_DIR, EVAL_LOG_DIR, TB_LOG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

ENV_ID          = "ALE/Breakout-v5"
SEED            = 42
TOTAL_TIMESTEPS = 500_000

# ── 10 Moderate Experiment Configs ────────────────────────────────────────────
#
#  All values stay in a moderate range:
#  lr          : 0.0001 - 0.0003  (no extremes)
#  gamma       : 0.97  - 0.99     (standard range)
#  batch_size  : 32    - 64       (moderate sizes)
#  eps_end     : 0.02  - 0.05     (not too greedy)
#  eps_fraction: 0.10  - 0.15     (moderate decay speed)
#
# ─────────────────────────────────────────────────────────────────────────────
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

# ── Consistent env wrapper stack ──────────────────────────────────────────────
def make_env(seed):
    """Same wrapper stack for train and eval — prevents type mismatch warning."""
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


# ── Single experiment runner ──────────────────────────────────────────────────
def run_experiment(cfg):
    exp_id       = cfg["exp"]
    tag          = f"exp{exp_id:02d}"
    exp_eval_dir = os.path.join(EVAL_LOG_DIR,   tag)
    exp_ckpt_dir = os.path.join(CHECKPOINT_DIR, tag)
    os.makedirs(exp_eval_dir, exist_ok=True)
    os.makedirs(exp_ckpt_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  Experiment {exp_id}/10  [{tag}]")
    print(f"  lr={cfg['lr']}  gamma={cfg['gamma']}  batch={cfg['batch']}")
    print(f"  eps_start={cfg['eps_start']}  eps_end={cfg['eps_end']}  eps_frac={cfg['eps_frac']}")
    print("=" * 60)

    # Both envs use same wrapper stack — no type mismatch
    train_env = make_env(SEED)
    eval_env  = make_env(SEED + 1)

    model = DQN(
        policy                  = "CnnPolicy",
        env                     = train_env,
        learning_rate           = cfg["lr"],
        gamma                   = cfg["gamma"],
        batch_size              = cfg["batch"],
        buffer_size             = 100_000,
        learning_starts         = 50_000,
        train_freq              = 4,
        target_update_interval  = 1000,
        exploration_initial_eps = cfg["eps_start"],
        exploration_final_eps   = cfg["eps_end"],
        exploration_fraction    = cfg["eps_frac"],
        optimize_memory_usage   = False,
        tensorboard_log         = TB_LOG_DIR,
        verbose                 = 0,
        seed                    = SEED,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 250_000,
        save_path   = exp_ckpt_dir,
        name_prefix = f"dqn_{tag}",
        verbose     = 0,
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
        tb_log_name         = tag,
        reset_num_timesteps = True,
        progress_bar        = True,
    )

    # Save final model immediately
    model_path = os.path.join(MODEL_DIR, f"dqn_{tag}")
    model.save(model_path)
    print(f"\n  Model saved → {model_path}.zip")

    # Load npz once and reuse — no double-load bug
    eval_npz    = os.path.join(exp_eval_dir, "evaluations.npz")
    mean_reward = 0.0
    std_reward  = 0.0
    noted       = "No eval data found."

    if os.path.exists(eval_npz):
        data        = np.load(eval_npz)
        timesteps   = data["timesteps"]
        all_results = data["results"]
        all_lengths = data["ep_lengths"]
        final       = all_results[-1]
        mean_reward = float(np.mean(final))
        std_reward  = float(np.std(final))
        trend       = all_results.mean(axis=1)

        if len(trend) > 1 and trend[-1] > trend[0]:
            noted = "Reward improving across training."
        elif len(trend) > 1 and trend[-1] < trend[0]:
            noted = "Reward declined — possible instability."
        else:
            noted = "Reward flat — may need more timesteps."

        # Save per-experiment reward log CSV
        exp_csv = os.path.join(LOG_DIR, f"{tag}_reward_log.csv")
        with open(exp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "mean_reward", "std_reward", "mean_ep_length"])
            for t, r, l in zip(timesteps, all_results, all_lengths):
                writer.writerow([
                    int(t),
                    round(float(np.mean(r)), 2),
                    round(float(np.std(r)),  2),
                    round(float(np.mean(l)), 1),
                ])
        print(f"  Log saved → {exp_csv}")

    print(f"  Exp {exp_id} done — Mean Reward: {mean_reward:.2f}  Std: {std_reward:.2f}")
    print(f"  Note: {noted}")

    # Immediate download link after each experiment
    zip_path = f"/kaggle/working/dqn_{tag}.zip"
    shutil.copy(f"{model_path}.zip", zip_path)
    print("  Download model:")
    display(FileLink(zip_path))
    print("  Download log:")
    display(FileLink(os.path.join(LOG_DIR, f"{tag}_reward_log.csv")))

    train_env.close()
    eval_env.close()

    return {
        "exp"        : exp_id,
        "lr"         : cfg["lr"],
        "gamma"      : cfg["gamma"],
        "batch"      : cfg["batch"],
        "eps_start"  : cfg["eps_start"],
        "eps_end"    : cfg["eps_end"],
        "eps_frac"   : cfg["eps_frac"],
        "mean_reward": round(mean_reward, 2),
        "std_reward" : round(std_reward,  2),
        "noted"      : noted,
    }


# ── Run All 10 Experiments ────────────────────────────────────────────────────
print("=" * 60)
print("  DQN Breakout — 10 Experiment Hyperparameter Sweep")
print(f"  Environment : {ENV_ID}")
print(f"  Timesteps   : {TOTAL_TIMESTEPS:,} per experiment")
print(f"  Policy      : CnnPolicy")
print("=" * 60)

summary_rows = []
for cfg in EXPERIMENTS:
    row = run_experiment(cfg)
    summary_rows.append(row)

# ── Save Master Summary CSV ───────────────────────────────────────────────────
summary_csv = os.path.join(LOG_DIR, "experiment_summary.csv")
fieldnames  = ["exp", "lr", "gamma", "batch",
               "eps_start", "eps_end", "eps_frac",
               "mean_reward", "std_reward", "noted"]

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

# ── Copy best model as dqn_model.zip for play.py ─────────────────────────────
best     = max(summary_rows, key=lambda x: x["mean_reward"])
best_tag = f"exp{best['exp']:02d}"
best_src = os.path.join(MODEL_DIR, f"dqn_{best_tag}.zip")
best_dst = os.path.join(MODEL_DIR, "dqn_model.zip")
if os.path.exists(best_src):
    shutil.copy(best_src, best_dst)
    print(f"\n  Best model copied → {best_dst}")

# ── Print Final Summary Table ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  All 10 Experiments Complete — Summary")
print("=" * 60)
print(f"  {'Exp':>4} {'lr':>8} {'gamma':>6} {'batch':>6} "
      f"{'eps_end':>8} {'eps_frac':>9} {'MeanRew':>9} {'Std':>6}")
print(f"  {'-'*4} {'-'*8} {'-'*6} {'-'*6} "
      f"{'-'*8} {'-'*9} {'-'*9} {'-'*6}")
for row in summary_rows:
    print(
        f"  {row['exp']:>4} "
        f"{row['lr']:>8} "
        f"{row['gamma']:>6} "
        f"{row['batch']:>6} "
        f"{row['eps_end']:>8} "
        f"{row['eps_frac']:>9} "
        f"{row['mean_reward']:>9.2f} "
        f"{row['std_reward']:>6.2f}"
    )

print(f"\n  Best Experiment : Exp {best['exp']}")
print(f"  Mean Reward     : {best['mean_reward']:.2f}")
print(f"  Config          : lr={best['lr']} gamma={best['gamma']} "
      f"batch={best['batch']} eps_end={best['eps_end']} eps_frac={best['eps_frac']}")

# ── Final download links ──────────────────────────────────────────────────────
print("\n  Download summary:")
display(FileLink(summary_csv))
print("  Download best model:")
display(FileLink(best_dst))

print("\n  Done! Run play.py next using dqn_model.zip.")
