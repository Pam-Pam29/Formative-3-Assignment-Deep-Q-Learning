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

MODEL_PATH = ".\models\dqn_exp02.zip"
ENV_ID     = "ALE/Breakout-v5"
N_EPISODES = 5
RECORD_DIR = "./videos/gameplay"
SEED       = 42

os.makedirs(RECORD_DIR, exist_ok=True)

print("=" * 60)
print("  DQN Breakout — Group Play")
print("  Model: models/dqn_exp02.zip")
print("  Victoria exp02 — Mean Reward 31.80")
print("=" * 60)
print(f"\n  Loading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"\n  ERROR: Model not found at {MODEL_PATH}")
    print("  Make sure models/dqn_exp02.zip exists in the repo root.")
    exit(1)

model = DQN.load(MODEL_PATH)
print("  Model loaded successfully ✓")
print(f"\n  Playing {N_EPISODES} episodes...")
print(f"  Live window + recording to {RECORD_DIR}/\n")

print(f"  {'Episode':>8} {'Reward':>10} {'Steps':>8} {'Duration':>10}")
print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*10}")

episode_rewards = []
episode_lengths = []

for ep in range(N_EPISODES):

    # ── Eval env — render_mode=human shows live game window ───────────────────
    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=SEED + ep,
                              env_kwargs={"render_mode": "human"})
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # ── Record env — raw env for saving video ─────────────────────────────────
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
        action, _ = model.predict(obs, deterministic=True)  # GreedyQPolicy
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()  # shows live game window

        # Mirror action to record env for video
        try:
            record_env.step(int(action[0]))
        except Exception:
            pass

        total_reward += float(reward[0])
        steps        += 1

        # True game over when lives reach 0
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

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  {'='*50}")
print(f"  Results over {N_EPISODES} episodes:")
print(f"  Mean Reward : {np.mean(episode_rewards):.2f}")
print(f"  Std Reward  : {np.std(episode_rewards):.2f}")
print(f"  Max Reward  : {np.max(episode_rewards):.2f}")
print(f"  Min Reward  : {np.min(episode_rewards):.2f}")
print(f"  Mean Steps  : {np.mean(episode_lengths):.0f}")
print(f"  Human baseline : 31.8")
print(f"  Random baseline: 1.7")
above = sum(1 for r in episode_rewards if r >= 31.8)
print(f"  Episodes above human baseline: {above} / {N_EPISODES}")
print(f"  {'='*50}")
print(f"\n  Videos saved → {RECORD_DIR}/")
print("\n  Done!")