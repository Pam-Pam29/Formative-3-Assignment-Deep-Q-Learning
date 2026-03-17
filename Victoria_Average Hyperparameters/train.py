"""
train.py — DQN Agent Training Script for Atari Breakout
Uses Stable Baselines3 + Gymnasium (ALE/Breakout-v5)

Usage:
    python train.py --exp 11          # Run experiment 11 (Member 2, average params)
    python train.py --all             # Run all 10 experiments for Member 2
    python train.py --best            # Train using the best known configuration
    python train.py --compare         # Compare CNNPolicy vs MLPPolicy (same hyperparams)
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# ─────────────────────────────────────────────
#  Member 2 — Average Hyperparameter Experiments (11–20)
# ─────────────────────────────────────────────
EXPERIMENTS = {
    11: dict(lr=0.0003, gamma=0.96, batch_size=32,  eps_start=1.0, eps_end=0.05, eps_decay=0.1),
    12: dict(lr=0.0003, gamma=0.97, batch_size=64,  eps_start=1.0, eps_end=0.05, eps_decay=0.1),
    13: dict(lr=0.0003, gamma=0.98, batch_size=64,  eps_start=1.0, eps_end=0.02, eps_decay=0.1),
    14: dict(lr=0.0005, gamma=0.96, batch_size=32,  eps_start=1.0, eps_end=0.05, eps_decay=0.15),
    15: dict(lr=0.0005, gamma=0.97, batch_size=64,  eps_start=1.0, eps_end=0.02, eps_decay=0.15),
    16: dict(lr=0.0005, gamma=0.98, batch_size=128, eps_start=1.0, eps_end=0.05, eps_decay=0.1),
    17: dict(lr=0.0003, gamma=0.97, batch_size=128, eps_start=1.0, eps_end=0.02, eps_decay=0.2),
    18: dict(lr=0.0005, gamma=0.98, batch_size=64,  eps_start=1.0, eps_end=0.05, eps_decay=0.2),
    19: dict(lr=0.0003, gamma=0.99, batch_size=32,  eps_start=1.0, eps_end=0.02, eps_decay=0.2),
    20: dict(lr=0.0005, gamma=0.99, batch_size=64,  eps_start=1.0, eps_end=0.01, eps_decay=0.2),
}

# Total timesteps per experiment — reduced for Kaggle RAM efficiency
TRAIN_TIMESTEPS = 100_000


def make_env(env_id: str = "ALE/Breakout-v5"):
    """Create and wrap the Atari Breakout environment."""
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    env = DummyVecEnv([_init])
    env = VecFrameStack(env, n_stack=4)
    return env


def build_dqn(env, params: dict, tensorboard_log: str = "./tb_logs/"):
    """
    Instantiate a DQN agent.

    Note on epsilon mapping:
        SB3 uses exploration_fraction (fraction of training where eps decays)
        and exploration_final_eps.  We map eps_decay → exploration_fraction.
    """
    model = DQN(
        policy="CnnPolicy",          # CNN is better for pixel-based Atari games
        env=env,
        learning_rate=params["lr"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        exploration_initial_eps=params["eps_start"],
        exploration_final_eps=params["eps_end"],
        exploration_fraction=params["eps_decay"],
        buffer_size=10_000,          # reduced from 100k to save RAM
        learning_starts=5_000,       # reduced from 10k
        train_freq=4,
        target_update_interval=1000,
        optimize_memory_usage=True,  # enabled to save RAM
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    return model


def train_experiment(exp_id: int, save_dir: str = "models"):
    """Train one experiment and save the model."""
    params = EXPERIMENTS[exp_id]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("eval_logs", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment {exp_id}")
    print(f"  lr={params['lr']}, gamma={params['gamma']}, "
          f"batch={params['batch_size']}, "
          f"eps: {params['eps_start']}→{params['eps_end']} "
          f"over {params['eps_decay']*100:.0f}% of training")
    print(f"{'='*60}\n")

    # ── Environments ──────────────────────────────────────────
    train_env = make_env()
    eval_env  = make_env()

    # ── Callbacks ─────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/exp_{exp_id}_best",
        log_path=f"eval_logs/exp_{exp_id}",
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"{save_dir}/exp_{exp_id}_checkpoints",
        name_prefix="dqn_breakout",
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_dqn(train_env, params,
                      tensorboard_log=f"./tb_logs/exp_{exp_id}/")

    # ── Training ──────────────────────────────────────────────
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=f"exp_{exp_id}",
    )

    # ── Save final model ──────────────────────────────────────
    save_path = os.path.join(save_dir, f"dqn_model_exp{exp_id}.zip")
    model.save(save_path)
    print(f"\n✓ Experiment {exp_id} saved → {save_path}")

    train_env.close()
    eval_env.close()
    return save_path


def train_best(save_dir: str = "models"):
    """
    Train using the best configuration found (Experiment 20 — highest gamma,
    moderate lr, good exploration decay for Breakout).
    Saves as dqn_model.zip (the file play.py expects).
    """
    best_exp = 20                        # Change after you run all experiments
    params   = EXPERIMENTS[best_exp]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BEST MODEL TRAINING  (based on Exp {best_exp})")
    print(f"  lr={params['lr']}, gamma={params['gamma']}, "
          f"batch={params['batch_size']}")
    print(f"{'='*60}\n")

    train_env = make_env()
    eval_env  = make_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model",
        log_path="eval_logs/best",
        eval_freq=25_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = build_dqn(train_env, params, tensorboard_log="./tb_logs/best/")
    model.learn(
        total_timesteps=200_000,
        callback=eval_callback,
        tb_log_name="best_model",
    )

    final_path = os.path.join(save_dir, "dqn_model.zip")
    model.save(final_path)
    print(f"\n✓ Best model saved → {final_path}")

    train_env.close()
    eval_env.close()


def compare_policies(save_dir: str = "models", timesteps: int = 100_000):
    """
    CNNPolicy vs MLPPolicy Comparison
    ──────────────────────────────────
    Runs BOTH policies under the SAME hyperparameters (Experiment 15)
    for a short 100k-step run, then evaluates each over 10 episodes.

    This answers the assignment requirement:
        "Compare MLPPolicy and CNNPolicy to see which performs better."

    Expected result for Breakout:
        CNNPolicy  → tracks ball/paddle via conv filters → higher reward
        MLPPolicy  → flattens pixels, loses spatial structure → near-random
    """
    # Use a mid-range experiment so the comparison is fair
    params = EXPERIMENTS[15]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Policy Comparison: CNNPolicy vs MLPPolicy")
    print(f"  Shared params: lr={params['lr']}, gamma={params['gamma']}, "
          f"batch={params['batch_size']}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    results = {}

    for policy_name in ["CnnPolicy", "MlpPolicy"]:
        print(f"\n── Training {policy_name} ──")

        train_env = make_env()

        model = DQN(
            policy=policy_name,
            env=train_env,
            learning_rate=params["lr"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            exploration_initial_eps=params["eps_start"],
            exploration_final_eps=params["eps_end"],
            exploration_fraction=params["eps_decay"],
            buffer_size=10_000,
            learning_starts=5_000,
            train_freq=4,
            target_update_interval=1000,
            optimize_memory_usage=True,
            verbose=1,
            tensorboard_log=f"./tb_logs/compare_{policy_name.lower()}/",
        )

        model.learn(
            total_timesteps=timesteps,
            tb_log_name=f"compare_{policy_name.lower()}",
        )

        # ── Evaluate over 10 episodes ──────────────────────────
        print(f"\n  Evaluating {policy_name} over 10 episodes...")
        eval_env = make_env()
        episode_rewards = []

        for ep in range(10):
            obs     = eval_env.reset()
            ep_rew  = 0.0
            done    = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)  # GreedyQPolicy
                obs, reward, done_arr, info = eval_env.step(action)
                ep_rew += reward[0]
                if done_arr[0]:
                    done = True
            episode_rewards.append(ep_rew)

        mean_rew = float(np.mean(episode_rewards))
        std_rew  = float(np.std(episode_rewards))
        results[policy_name] = {"mean": mean_rew, "std": std_rew,
                                 "all": episode_rewards}

        print(f"  {policy_name}: mean={mean_rew:.2f}, std={std_rew:.2f}")

        # Save model so it can be inspected later
        model.save(os.path.join(save_dir, f"compare_{policy_name.lower()}.zip"))

        train_env.close()
        eval_env.close()

    # ── Print comparison table ─────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPARISON RESULT")
    print(f"{'='*60}")
    print(f"  {'Policy':<14} {'Mean Reward':>12}  {'Std':>8}  {'Decision'}")
    print(f"  {'-'*50}")
    winner = max(results, key=lambda k: results[k]["mean"])
    for pol, r in results.items():
        flag = " ← WINNER" if pol == winner else ""
        print(f"  {pol:<14} {r['mean']:>12.2f}  {r['std']:>8.2f}{flag}")
    print(f"\n  Conclusion: {winner} is better for Atari Breakout.")
    print(f"  Reason: Breakout requires tracking ball position across frames.")
    print(f"  CNNPolicy learns spatial filters; MLPPolicy sees only raw numbers.")
    print(f"{'='*60}\n")

    return results


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Atari Breakout")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp",  type=int, choices=list(EXPERIMENTS.keys()),
                       help="Run a single experiment by ID (11–20)")
    group.add_argument("--all",  action="store_true",
                       help="Run all 10 Member-2 experiments sequentially")
    group.add_argument("--best", action="store_true",
                       help="Train the best configuration (saves dqn_model.zip)")
    group.add_argument("--compare", action="store_true",
                       help="Compare CNNPolicy vs MLPPolicy (100k steps each)")
    args = parser.parse_args()

    if args.exp:
        train_experiment(args.exp)

    elif args.all:
        results = {}
        for exp_id in EXPERIMENTS:
            path = train_experiment(exp_id)
            results[exp_id] = path
        print("\n\n── All experiments complete ──")
        for eid, path in results.items():
            p = EXPERIMENTS[eid]
            print(f"  Exp {eid}: lr={p['lr']}, γ={p['gamma']}, "
                  f"batch={p['batch_size']} → {path}")

    elif args.best:
        train_best()

    elif args.compare:
        compare_policies()
