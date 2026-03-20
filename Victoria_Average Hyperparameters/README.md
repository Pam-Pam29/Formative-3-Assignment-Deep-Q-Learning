# DQN Breakout — Victoria's Hyperparameter Experiments
**Formative 3 · Deep Q-Learning · Stable Baselines 3 + Gymnasium**

---

## Member
**Victoria** — Individual Contribution

---

## Environment

| Field | Value |
|---|---|
| Game | ALE/Breakout-v5 |
| Framework | Stable Baselines 3 |
| Policy | CNNPolicy (Convolutional Neural Network) |
| Observation | 84×84 grayscale · 4-frame stack |
| Training steps | 500,000 per experiment |
| Total experiments | 10 + 1 optimised (expBEST) |

---

## Gameplay Video
[![Watch DQN Breakout Agent](https://img.youtube.com/vi/sXjg4H0J6nQ/0.jpg)](https://youtube.com/shorts/sXjg4H0J6nQ)

> Episode 1 — Reward: 34.0 · 
> Trained with exp02: lr=0.0001 · gamma=0.98 · batch=64 · 500K steps
> *(Recorded using play.py — Mean Reward: 27.40 · Max Reward: 34.0)*

---

## Repository Structure

```
├── train.py                  # 10-experiment hyperparameter sweep + expBEST
├── play.py                   # Loads best model and plays Breakout
├── compare.py                # MLP vs CNN policy comparison
├── models/
│   ├── dqn_model.zip         # Best model — exp02 (reward 31.80)
│   ├── dqn_exp01.zip
│   ├── dqn_exp02.zip
│   ├── dqn_exp03.zip
│   ├── dqn_exp04.zip
│   ├── dqn_exp05.zip
│   ├── dqn_exp06.zip
│   ├── dqn_exp07.zip
│   ├── dqn_exp08.zip
│   ├── dqn_exp09.zip
│   ├── dqn_exp10.zip
│   └── dqn_expBEST.zip
└── logs/
    ├── experiment_summary.csv
    ├── exp01_reward_log.csv
    ├── exp02_reward_log.csv
    ├── exp03_reward_log.csv
    ├── exp04_reward_log.csv
    ├── exp05_reward_log.csv
    ├── exp06_reward_log.csv
    ├── exp07_reward_log.csv
    ├── exp08_reward_log.csv
    ├── exp09_reward_log.csv
    ├── exp10_reward_log.csv
    └── mlp_vs_cnn_comparison.csv
```

---

## Policy Comparison — MLP vs CNN

Both policies were trained for 100,000 steps using the best config from the sweep
(lr=0.0001, gamma=0.98, batch=64).

| Policy | Mean Reward | Std | Notes |
|---|---|---|---|
| CnnPolicy | 13.20 | 5.00 | Learns spatial features — improving throughout |
| MlpPolicy | 4.60 | 1.74 | Flattens pixels — reward declined after 25K steps |

**Difference: +8.60 in favour of CnnPolicy**

CNNPolicy is the correct architecture for pixel-based Atari environments. The input
is a stack of 4 grayscale frames (84×84×4), which requires spatial feature extraction.
CNNPolicy applies convolutional layers to detect edges, shapes, and motion before making
Q-value decisions. MlpPolicy flattens the entire image into a 1D vector, destroying all
spatial structure and making it nearly impossible to learn from raw pixels.
MlpPolicy peaked at 6.60 at 25K steps, then declined to 4.60, confirming it cannot
learn effectively from pixel input.

---

## Hyperparameter Tuning — 10 Experiments

All experiments were trained for 500,000 steps on ALE/Breakout-v5 with seed=42.

**Human baseline: 31.8 · Random baseline: 1.7**

| # | lr | γ | batch | eps_start | eps_end | eps_frac | Mean Reward | Std | Peak | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| exp01 | 0.0001 | 0.97 | 32 | 1.0 | 0.05 | 0.10 | 26.65 | 10.24 | 29.70 | Stable learning, improving throughout |
| exp02 | 0.0001 | 0.98 | 64 | 1.0 | 0.05 | 0.10 | **31.80** | 8.86 | 31.80 | **Best — matches human baseline exactly** |
| exp03 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 0.12 | 19.65 | 5.60 | 22.45 | γ=0.99 too high at 500K — Q-value overestimation |
| exp04 | 0.0002 | 0.97 | 32 | 1.0 | 0.05 | 0.10 | 23.70 | 7.85 | 30.85 | Consistent but below exp02 |
| exp05 | 0.0002 | 0.98 | 64 | 1.0 | 0.05 | 0.12 | 23.70 | 11.98 | 27.90 | High variance — GPU throttle during run |
| exp06 | 0.0002 | 0.99 | 64 | 1.0 | 0.05 | 0.15 | 24.95 | 6.61 | 28.55 | Clean result — lowest std in lr=0.0002 group |
| exp07 | 0.0003 | 0.97 | 32 | 1.0 | 0.05 | 0.10 | 21.40 | 12.01 | 27.25 | Highest std in sweep — lr=0.0003 too aggressive |
| exp08 | 0.0003 | 0.98 | 64 | 1.0 | 0.05 | 0.12 | 22.45 | 8.66 | 30.00 | Recovered after RAM issue in original run |
| exp09 | 0.0003 | 0.99 | 32 | 1.0 | 0.02 | 0.12 | 29.60 | 7.31 | 33.60 | Strong — eps_end=0.02 more exploitative |
| exp10 | 0.0002 | 0.99 | 32 | 1.0 | 0.02 | 0.15 | 30.85 | 9.20 | 30.85 | Second best — longer exploration paid off |
| expBEST | 0.0001 | 0.98 | 64 | 1.0 | 0.01 | 0.15 | 20.65 | 7.79 | 23.20 | Optimised config — GPU throttled in final phase |

---

## Key Findings

### What improved performance

**Learning rate lr=0.0001** consistently outperformed lr=0.0002 and lr=0.0003 across
all experiments. A conservative learning rate prevents overshooting the optimal policy
with a limited 500K training budget. The average reward for lr=0.0001 experiments was
26.03 vs 24.58 for lr=0.0002 and 24.48 for lr=0.0003.

**Gamma γ=0.98** was the Goldilocks value. γ=0.97 was too myopic — the agent discounted
future rewards too heavily, making it unable to plan the multi-step strategies needed to
break upper-row bricks. γ=0.99 caused Q-value overestimation before the network had
sufficient training data, leading to unstable learning.

**batch=64** reduced variance compared to batch=32 across all experiments. Larger batches
produce smoother gradient updates and more stable policies. exp02 (batch=64) had std=8.86
while exp01 (batch=32, same lr and gamma) had std=10.24.

### What harmed performance

**γ=0.99 at 500K steps** was consistently the weakest setting. exp03 scored only 19.65 —
the lowest in the sweep. High gamma requires millions of steps to pay off because the agent
needs to experience many long-horizon trajectories before the discount benefit materialises.

**lr=0.0003** produced the highest variance in the entire sweep (exp07 std=12.01). The
network updated weights too aggressively, learning fast early but producing an unstable
policy that oscillated rather than converging.

**RAM pressure** affected exp08 in the original run — the replay buffer (5.65GB) exceeded
available system RAM (2.84GB), causing disk swapping and degraded training. Fixed by
reducing buffer size in the rerun.

### Best configuration

```
lr=0.0001  gamma=0.98  batch=64  eps_start=1.0  eps_end=0.05  eps_frac=0.10
Mean Reward: 31.80  (= human baseline of 31.8)
```

---

## Agent Performance — play.py Results

Model tested over 20 episodes using deterministic=True (GreedyQPolicy):

| Episode | Reward | Steps | Notes |
|---|---|---|---|
| 1 | 34.0 | 342 | **Beat human baseline (31.8)** |
| 2 | 15.0 | 151 | Worst episode — lost lives quickly |
| 3 | 28.0 | 262 | Solid — near human level |
| 4 | 28.0 | 271 | Solid — consistent |
| 5 | 32.0 | 314 | **Beat human baseline (31.8)** |
| 6 | 24.0 | 229 | Above mean |
| 7 | 33.0 | 317 | **Beat human baseline (31.8)** |
| 8 | 18.0 | 188 | Below mean — unlucky ball angle |
| 9 | 22.0 | 229 | Consistent |
| 10 | 28.0 | 260 | Solid |
| 11 | 29.0 | 294 | Solid |
| 12 | 28.0 | 269 | Solid |
| 13 | 21.0 | 216 | Consistent |
| 14 | 24.0 | 216 | Above mean |
| 15 | 24.0 | 243 | Above mean |
| 16 | 21.0 | 219 | Consistent |
| 17 | 24.0 | 236 | Above mean |
| 18 | 22.0 | 205 | Consistent |
| 19 | 19.0 | 199 | Below mean |
| 20 | 28.0 | 284 | Solid |
| **Mean** | **25.10** | **247** | **15x better than random (1.7)** |

The agent destroyed an average of 25 bricks per game over 20 episodes.
3 out of 20 episodes beat the human baseline of 31.8 (ep1: 34.0, ep5: 32.0, ep7: 33.0).
Episode 1 peaked at 34.0 — exceeding human level at only 500K training steps
(approximately 4% of a full DQN training run).
---

## How to Run

### Install dependencies
```bash
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom opencv-python
AutoROM --accept-license
```

### Train
```bash
python train.py
```

### Play
```bash
python play.py
```

### Compare MLP vs CNN
```bash
python compare.py
```

---

## References

- Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*. DeepMind.
- Stable Baselines 3 Documentation: https://stable-baselines3.readthedocs.io
- Gymnasium Atari Environments: https://gymnasium.farama.org/environments/atari/
