"""
================================================
  play.py — DQN Breakout
  Loads the best trained model and plays
  Records gameplay video
  Stable Baselines 3 + Gymnasium
================================================
"""

import os
import time
import numpy as np
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from gymnasium.wrappers import RecordVideo

gym.register_envs(ale_py)

MODEL_PATH = "./Best Model/dqn_model.zip"
ENV_ID     = "ALE/Breakout-v5"
N_EPISODES = 20             
RECORD_DIR = "./videos/gameplay"
SEED       = 42

os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs("./models", exist_ok=True)

print("=" * 60)
print("  DQN Breakout — Play Mode")
print("=" * 60)
print(f"\n  Loading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"\n  ERROR: Model not found at {MODEL_PATH}")
    exit(1)

model = DQN.load(MODEL_PATH)
print("  Model loaded successfully ✓")
print(f"\n  Playing {N_EPISODES} episodes...")
print(f"  Recording to {RECORD_DIR}/\n")

print(f"  {'Episode':>8} {'Reward':>10} {'Steps':>8} {'Duration':>10}")
print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*10}")

episode_rewards = []
episode_lengths = []

for ep in range(N_EPISODES):

    # ── Eval env — exact training wrapper stack
    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=SEED + ep)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # ── Record env — raw env for video only 
    record_env = RecordVideo(
        gym.make(ENV_ID, render_mode="rgb_array"),
        video_folder    = RECORD_DIR,
        episode_trigger = lambda e: True,
        name_prefix     = f"breakout_ep{ep+1}",
    )
    record_env.reset(seed=SEED + ep)

    obs          = eval_env.reset()
    total_reward = 0.0
    steps        = 0
    lives        = None
    start        = time.time()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

        # Mirror to record env for video
        try:
            record_env.step(int(action[0]))
        except Exception:
            pass

        total_reward += float(reward[0])
        steps        += 1

        # Get lives — true game over when lives reach 0
        current_lives = info[0].get("lives", None)
        if lives is None:
            lives = current_lives
        if current_lives is not None and current_lives == 0:
            break

        # Safety cap
        if steps >= 5000:
            break

    duration = time.time() - start
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

    eval_env.close()
    record_env.close()

    print(f"  {ep+1:>8} {total_reward:>10.1f} {steps:>8} {duration:>9.1f}s")

print(f"\n  {'='*50}")
print(f"  Results over {N_EPISODES} episodes:")
print(f"  Mean Reward : {np.mean(episode_rewards):.2f}")
print(f"  Std Reward  : {np.std(episode_rewards):.2f}")
print(f"  Max Reward  : {np.max(episode_rewards):.2f}")
print(f"  Mean Steps  : {np.mean(episode_lengths):.0f}")
print(f"  {'='*50}")
print(f"\n  Videos saved → {RECORD_DIR}/")
print("\n  Done!")
