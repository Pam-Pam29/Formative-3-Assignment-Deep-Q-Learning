"""
================================================
  train.py — DQN Breakout
  Group Assignment — Unified Training Script
  Members: Erneste, Victoria, Pretty
  Stable Baselines 3 + Gymnasium
================================================
"""

import os
import csv
import json
import time
import argparse
import warnings
import logging
import numpy as np

# Suppress noisy CUDA / TensorFlow registration warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

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

ENV_ID = "ALE/Breakout-v5"
SEED   = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train one DQN Breakout experiment")
    parser.add_argument("--member",          type=str,   required=True)
    parser.add_argument("--experiment",      type=str,   required=True)
    parser.add_argument("--lr",              type=float, required=True)
    parser.add_argument("--gamma",           type=float, required=True)
    parser.add_argument("--batch",           type=int,   required=True)
    parser.add_argument("--eps-start",       type=float, default=1.0)
    parser.add_argument("--eps-end",         type=float, required=True)
    parser.add_argument("--eps-frac",        type=float, required=True)
    parser.add_argument("--total-timesteps", type=int,   default=100_000)
    parser.add_argument("--buffer-size",     type=int,   default=100_000)
    return parser.parse_args()


def make_env(seed):
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


def main():
    args = parse_args()
    member  = args.member
    exp_tag = args.experiment
    tag     = exp_tag

    base_dir     = os.path.join("results", member)
    model_dir    = os.path.join(base_dir, "models")
    log_dir      = os.path.join(base_dir, "logs")
    exp_eval_dir = os.path.join(base_dir, exp_tag, "eval_logs")
    exp_ckpt_dir = os.path.join(base_dir, exp_tag, "checkpoints")

    for d in [model_dir, log_dir, exp_eval_dir, exp_ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    config = {
        "member": member, "experiment": exp_tag, "env_id": ENV_ID,
        "lr": args.lr, "gamma": args.gamma, "batch": args.batch,
        "eps_start": args.eps_start, "eps_end": args.eps_end,
        "eps_frac": args.eps_frac,
        "total_timesteps": args.total_timesteps,
        "buffer_size": args.buffer_size,
    }
    with open(os.path.join(base_dir, exp_tag, f"{exp_tag}_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  [{member}]  {exp_tag}")
    print(f"  lr={args.lr}  gamma={args.gamma}  batch={args.batch}")
    print(f"  eps_end={args.eps_end}  eps_frac={args.eps_frac}")
    print("=" * 65)

    train_env = make_env(SEED)
    eval_env  = make_env(SEED + 1)

    model = DQN(
        policy="CnnPolicy", env=train_env,
        learning_rate=args.lr, gamma=args.gamma, batch_size=args.batch,
        buffer_size=args.buffer_size, learning_starts=10_000,
        train_freq=4, target_update_interval=1_000,
        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        exploration_fraction=args.eps_frac,
        verbose=1, seed=SEED,
    )
    # NOTE: optimize_memory_usage is NOT used here.
    # It is incompatible with handle_timeout_termination=True in SB3
    # and causes: ValueError: ReplayBuffer does not support
    # optimize_memory_usage=True and handle_timeout_termination=True simultaneously.

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000, save_path=exp_ckpt_dir,
        name_prefix=f"dqn_{tag}", verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=exp_eval_dir,
        log_path=exp_eval_dir, eval_freq=20_000,
        n_eval_episodes=5, deterministic=True, verbose=0,
    )

    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([checkpoint_cb, eval_cb]),
        reset_num_timesteps=True, progress_bar=False,
    )
    train_minutes = (time.time() - start_time) / 60.0

    final_model_path = os.path.join(model_dir, f"dqn_{tag}")
    model.save(final_model_path)

    eval_npz    = os.path.join(exp_eval_dir, "evaluations.npz")
    mean_reward = 0.0
    std_reward  = 0.0
    noted       = "No eval data."
    exp_csv     = os.path.join(log_dir, f"{tag}_reward_log.csv")

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
            "Reward improving across training."            if trend[-1] > trend[0]
            else "Reward declined — possible instability." if trend[-1] < trend[0]
            else "Reward flat — may need more timesteps."
        )
        with open(exp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "mean_reward", "std_reward", "mean_ep_length"])
            for t, r, l in zip(timesteps, all_results, all_lengths):
                writer.writerow([int(t), round(float(np.mean(r)), 2),
                                 round(float(np.std(r)), 2),
                                 round(float(np.mean(l)), 1)])

    eval_summary = {
        "member": member, "experiment": exp_tag, "env_id": ENV_ID,
        "mean_reward": round(mean_reward, 2), "std_reward": round(std_reward, 2),
        "train_minutes": round(train_minutes, 2), "noted": noted,
        "model_path": f"{final_model_path}.zip",
        "best_model_path": os.path.join(exp_eval_dir, "best_model.zip"),
        "training_csv_path": exp_csv,
        **config,
    }
    eval_json_path = os.path.join(base_dir, exp_tag, f"{exp_tag}_eval.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    train_env.close()
    eval_env.close()

    print(f"\n  ✓ {tag}  Mean Reward: {mean_reward:.2f}  Std: {std_reward:.2f}")
    print(f"  Train minutes: {train_minutes:.2f}")


if __name__ == "__main__":
    main()
