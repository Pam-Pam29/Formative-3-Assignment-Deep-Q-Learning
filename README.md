# DQN Breakout — Group Hyperparameter Experiment
**Formative 3 · Deep Q-Learning · Stable Baselines 3 + Gymnasium**

---

## Group Members

| Member | Role | Hyperparameter Range |
|---|---|---|
| **Victoria Fakunle** | Average Hyperparameters | lr=0.0001–0.0003, γ=0.97–0.99, batch=32–64 |
| **Pretty Ntakirutimana** | Lower Hyperparameters | lr=0.00001–0.00005, γ=0.90–0.95, batch=16–32 |
| **Erneste** | Higher Hyperparameters | lr=0.0005–0.001, γ=0.990–0.999, batch=128–256 |

---

## Environment

| Field | Value |
|---|---|
| Game | ALE/Breakout-v5 |
| Framework | Stable Baselines 3 |
| Policy | CnnPolicy (Convolutional Neural Network) |
| Observation | 84×84 grayscale · 4-frame stack |
| Training steps | 500,000 per experiment |
| Total experiments | 30 (10 per member) |

---

## Gameplay Video

[![Watch DQN Breakout Agent](https://img.youtube.com/vi/sXjg4H0J6nQ/0.jpg)](https://youtube.com/shorts/sXjg4H0J6nQ)

> Best episode — Reward: 34.0 · Beats human baseline (31.8)
> Recorded using play.py with Victoria's exp02 model (group best · 31.80 mean reward)

---

## Repository Structure

```
Formative-3-Assignment-Deep-Q-Learning/
│
├── README.md                                # This file — group overview
├── play.py                                  # Shared — loads group best model
├── compare.py                               # Shared — MLP vs CNN comparison
│
├── Best Model/
│   └── dqn_model.zip                        # Group best model — Victoria exp02
│
├── videos/
│   └── gameplay/
│       └── breakout_ep01–20.mp4             # Recorded by root play.py
│
├── Victoria_Average Hyperparameters/
│   ├── train.py                             # Victoria's 10 experiments
│   ├── play.py                              # Victoria's individual play script
│   ├── Best Model/
│   │   └── dqn_model.zip                    # Victoria's best model
│   ├── models/
│   │   └── dqn_exp01–10.zip + dqn_expBEST.zip
│   ├── Logs/
│   │   └── experiment_summary.csv + exp01–10_reward_log.csv + mlp_vs_cnn_comparison.csv
│   └── Gameplay Videos/
│       └── breakout_ep1-episode-0.mp4
│
├── Pretty_Lower Hyperparameters/
│   ├── train.py                             # Pretty's 10 experiments
│   ├── play.py                              # Pretty's individual play script
│   ├── models/
│   │   └── dqn_model.zip + dqn_exp01–10.zip
│   └── logs/
│       └── experiment_summary.csv + exp01–10_reward_log.csv
│
└── Erneste_Higher Hyperparameters/
    ├── train.py                             # Erneste's 10 experiments
    ├── play.py                              # Erneste's individual play script
    ├── models/
    │   └── dqn_model.zip + dqn_exp01–10.zip
    └── logs/
        └── experiment_summary.csv + exp01–10_reward_log.csv
```

---

## Policy Comparison — MLP vs CNN

CnnPolicy is the correct architecture for pixel-based Atari environments. The input is a stack of 4 grayscale frames (84×84×4), which requires spatial feature extraction. CnnPolicy applies convolutional layers to detect edges, shapes, and motion before making Q-value decisions. MlpPolicy flattens the entire image into a 1D vector, destroying all spatial structure and making it nearly impossible to learn from raw pixels.

Run `compare.py` to reproduce these results.

| Policy | Mean Reward | Std | Notes |
|---|---|---|---|
| CnnPolicy | 13.20 | 5.00 | Standard for pixel-based RL — improving throughout |
| MlpPolicy | 4.60 | 1.74 | Flattens pixels — reward declined after 25K steps |

Both policies were trained for 100,000 steps using the same hyperparameters (lr=0.0001, gamma=0.98, batch=64). CnnPolicy scored 13.20 vs MlpPolicy's 4.60 — a difference of +8.60. MlpPolicy peaked at 25K steps (6.60) then declined to 4.60, confirming it cannot learn effectively from raw pixels. CnnPolicy improved consistently across all 4 checkpoints, reaching 13.90 at 75K before settling at 13.20.

---

## Hyperparameter Tuning — All 30 Experiments

**Human baseline: 31.8 · Random baseline: 1.7**

---

### Victoria Fakunle — Average Hyperparameters

| # | lr | γ | batch | eps_start | eps_end | eps_frac | Mean Reward | Std | Peak | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| exp01 | 0.0001 | 0.97 | 32 | 1.0 | 0.05 | 0.10 | 26.65 | 10.24 | 29.70 | Stable learning, improving throughout |
| exp02 | 0.0001 | 0.98 | 64 | 1.0 | 0.05 | 0.10 | **31.80** | 8.86 | 31.80 | **Best overall — matches human baseline** |
| exp03 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 0.12 | 19.65 | 5.60 | 22.45 | γ=0.99 causes Q-value overestimation at 500K |
| exp04 | 0.0002 | 0.97 | 32 | 1.0 | 0.05 | 0.10 | 23.70 | 7.85 | 30.85 | Consistent but below exp02 |
| exp05 | 0.0002 | 0.98 | 64 | 1.0 | 0.05 | 0.12 | 23.70 | 11.98 | 27.90 | High variance — GPU throttle during run |
| exp06 | 0.0002 | 0.99 | 64 | 1.0 | 0.05 | 0.15 | 24.95 | 6.61 | 28.55 | Clean result, lowest std for lr=0.0002 group |
| exp07 | 0.0003 | 0.97 | 32 | 1.0 | 0.05 | 0.10 | 21.40 | 12.01 | 27.25 | Highest std in sweep — lr too aggressive |
| exp08 | 0.0003 | 0.98 | 64 | 1.0 | 0.05 | 0.12 | 22.45 | 8.66 | 30.00 | Recovered after RAM issue in original run |
| exp09 | 0.0003 | 0.99 | 32 | 1.0 | 0.02 | 0.12 | 29.60 | 7.31 | 33.60 | Strong — eps_end=0.02 more exploitative |
| exp10 | 0.0002 | 0.99 | 32 | 1.0 | 0.02 | 0.15 | 30.85 | 9.20 | 30.85 | Second best — longer exploration paid off |

**Victoria's best:** exp02 — Mean Reward **31.80**

---

### Pretty Ntakirutimana — Lower Hyperparameters

> Note: exp01–08 and exp10 ran for 100,000 steps. exp09 ran for 500,000 steps.

| # | lr | γ | batch | eps_start | eps_end | eps_frac | Steps | Mean Reward | Peak | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| exp01 | 0.00001 | 0.90 | 16 | 1.0 | 0.01 | 0.05 | 100K | 2.40 | 2.40 | Extremely slow learning — lr too low for 100K |
| exp02 | 0.00001 | 0.92 | 16 | 1.0 | 0.01 | 0.05 | 100K | 2.60 | 5.00 | Initial promise at 40K then significant drop |
| exp03 | 0.00001 | 0.95 | 32 | 1.0 | 0.01 | 0.08 | 100K | 5.40 | 6.20 | Strong start (6.2 at 20K) but highly unstable |
| exp04 | 0.00003 | 0.90 | 16 | 1.0 | 0.01 | 0.05 | 100K | 8.40 | 8.40 | Consistent slow improvement — peak at end |
| exp05 | 0.00003 | 0.92 | 32 | 1.0 | 0.02 | 0.08 | 100K | 7.00 | 11.20 | High peak at 80K then degraded final 20K |
| exp06 | 0.00003 | 0.95 | 32 | 1.0 | 0.02 | 0.08 | 100K | 12.60 | 12.60 | Strong final performance — notable dip at 80K |
| exp07 | 0.00005 | 0.90 | 16 | 1.0 | 0.01 | 0.05 | 100K | 4.00 | 12.60 | Highest mid-run potential then catastrophic collapse |
| exp08 | 0.00005 | 0.92 | 32 | 1.0 | 0.02 | 0.08 | 100K | 10.40 | 10.40 | Very steady and stable growth throughout |
| exp09 | 0.00005 | 0.95 | 32 | 1.0 | 0.03 | 0.10 | **500K** | **23.80** | **23.80** | **Best — sustained learning over longer duration** |
| exp10 | 0.00003 | 0.95 | 16 | 1.0 | 0.02 | 0.10 | 100K | 8.80 | 8.80 | Modest but consistent growth similar to exp08 |

**Pretty's best:** exp09 — Mean Reward **23.80**

---

### Erneste — Higher Hyperparameters

| # | lr | γ | batch | eps_start | eps_end | eps_frac | Mean Reward | Std | Peak | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| exp01 | 0.0005 | 0.990 | 128 | 1.0 | 0.10 | 0.15 | 26.70 | 8.16 | 26.70 | Reward improving across training |
| exp02 | 0.0005 | 0.995 | 128 | 1.0 | 0.10 | 0.20 | 23.85 | 8.90 | 24.80 | Reward improving across training |
| exp03 | 0.0005 | 0.999 | 256 | 1.0 | 0.10 | 0.20 | 23.45 | 7.52 | 23.65 | Reward flat — γ=0.999 too high at 500K |
| exp04 | 0.0007 | 0.990 | 128 | 1.0 | 0.10 | 0.15 | 23.45 | 6.74 | 29.25 | Peaked early at 200K then declined |
| exp05 | 0.0007 | 0.995 | 256 | 1.0 | 0.10 | 0.20 | 23.25 | 6.68 | 24.75 | Reward flat — may need more timesteps |
| exp06 | 0.0007 | 0.999 | 256 | 1.0 | 0.15 | 0.25 | 23.70 | 7.41 | 24.45 | Reward improving across training |
| exp07 | 0.0010 | 0.990 | 128 | 1.0 | 0.10 | 0.15 | 27.05 | 5.54 | 27.05 | Reward improving — lowest std in sweep |
| exp08 | 0.0010 | 0.995 | 256 | 1.0 | 0.15 | 0.25 | **27.80** | 7.51 | 30.90 | **Best — high batch stabilised aggressive lr** |
| exp09 | 0.0010 | 0.999 | 256 | 1.0 | 0.15 | 0.30 | 18.30 | 6.72 | 26.30 | Policy collapsed after 350K — γ=0.999 instability |
| exp10 | 0.0007 | 0.999 | 128 | 1.0 | 0.10 | 0.25 | 21.30 | 8.20 | 28.25 | Peaked late at 450K then declined |

**Erneste's best:** exp08 — Mean Reward **27.80**

---

## Best Model Selection

| Member | Best Exp | Mean Reward | Model Location |
|---|---|---|---|
| Victoria Fakunle | exp02 | **31.80** | Victoria_Average Hyperparameters/Best Model/ |
| Pretty Ntakirutimana | exp09 | 23.80 | Pretty_Lower Hyperparameters/models/ |
| Erneste | exp08 | 27.80 | Erneste_Higher Hyperparameters/models/ |
| **Group best** | **Victoria exp02** | **31.80** | **Best_Model/dqn_model.zip** |

> Victoria's exp02 is the group best — copied to Best_Model/dqn_model.zip and used by the shared play.py.

---

## Key Findings

### Victoria Fakunle — Average Hyperparameters

**What improved performance:**
- `lr=0.0001` consistently outperformed higher learning rates. Conservative lr prevents overshooting at a 500K step budget.
- `γ=0.98` was the sweet spot. γ=0.97 was too myopic for Breakout's multi-step rally rewards; γ=0.99 caused Q-value overestimation before the network had enough data.
- `batch=64` reduced variance vs batch=32. Larger batches produce smoother gradient updates and more stable policies.

**What harmed performance:**
- `γ=0.99 at 500K steps` — exp03 scored only 19.65. High gamma requires millions of training steps to pay off.
- `lr=0.0003` — exp07 std=12.01, highest in the sweep. Fast learning but unstable policy.
- RAM pressure during exp08's original run caused buffer swapping which degraded training quality.

**Best config:** `lr=0.0001, γ=0.98, batch=64, eps_end=0.05, eps_frac=0.10` → Mean Reward **31.80**

---

### Pretty Ntakirutimana — Lower Hyperparameters

**What improved performance:**
- Training duration was the single biggest factor. exp09 ran for 500K steps and scored 23.80 — nearly double the best 100K result (exp06: 12.60). With very low learning rates, the agent simply needs more time to converge.
- `lr=0.00005` with `γ=0.95` was the most effective combination — the highest lr in the lower range combined with the highest gamma gave the agent enough speed and long-term planning.
- `batch=32` provided more stable updates than `batch=16`. All experiments with batch=32 outperformed their batch=16 equivalents at the same lr.

**What harmed performance:**
- `lr=0.00001` was too slow for any useful learning in 100K steps — exp01 and exp02 scored only 2.40 and 2.60, barely above random.
- `batch=16` produced very noisy gradients — exp07 collapsed from 12.60 at 60K to 4.00 at 100K, the most dramatic instability in the sweep.
- 100K steps was insufficient for the lower hyperparameter range — most experiments were still improving when training stopped.

**Best config:** `lr=0.00005, γ=0.95, batch=32, eps_end=0.03, eps_frac=0.10` → Mean Reward **23.80** (500K steps)

---

### Erneste — Higher Hyperparameters

**What improved performance:**
- `lr=0.001` was the best learning rate — aggressive enough to make progress with large batches and high gamma. exp07 and exp08 (both lr=0.001) were the top two results.
- `batch=256` with `lr=0.001` stabilised training — larger batches smoothed noisy gradients from aggressive learning rates.
- `γ=0.995` balanced long-horizon planning without overestimation — outperforming both γ=0.990 and γ=0.999 at the same lr.

**What harmed performance:**
- `γ=0.999` consistently caused instability. exp09 peaked at 26.30 then collapsed to 18.30 — policy diverged in the second half of training.
- Several experiments peaked early then declined — exp04 hit 29.25 at 200K but dropped to 23.45 by 500K, showing lr=0.0007 overshoots with high gamma.

**Best config:** `lr=0.001, γ=0.995, batch=256, eps_end=0.15, eps_frac=0.25` → Mean Reward **27.80** (Peak 30.90)

---

## Agent Performance — play.py Results

Model tested over 20 episodes using Victoria's exp02 (group best), deterministic=True (GreedyQPolicy):

| Episode | Reward | Steps | Notes |
|---|---|---|---|
| 1 | 34.0 | 342 | **Beat human baseline (31.8)** |
| 2 | 15.0 | 151 | Below mean — worst episode, lost lives quickly |
| 3 | 28.0 | 262 | Above mean — near human level |
| 4 | 28.0 | 271 | Above mean — consistent |
| 5 | 32.0 | 314 | **Beat human baseline (31.8)** |
| 6 | 24.0 | 229 | Below mean |
| 7 | 33.0 | 317 | **Beat human baseline (31.8)** |
| 8 | 18.0 | 188 | Below mean — unlucky ball angle |
| 9 | 22.0 | 229 | Below mean |
| 10 | 28.0 | 260 | Above mean |
| 11 | 29.0 | 294 | Above mean |
| 12 | 28.0 | 269 | Above mean |
| 13 | 21.0 | 216 | Below mean |
| 14 | 24.0 | 216 | Below mean |
| 15 | 24.0 | 243 | Below mean |
| 16 | 21.0 | 219 | Below mean |
| 17 | 24.0 | 236 | Below mean |
| 18 | 22.0 | 205 | Below mean |
| 19 | 19.0 | 199 | Below mean |
| 20 | 28.0 | 284 | Above mean |
| **Mean** | **25.10** | **247** | **15x better than random (1.7)** |

3 out of 20 episodes beat the human baseline of 31.8 (ep1: 34.0, ep5: 32.0, ep7: 33.0).
The agent destroyed an average of 25 bricks per game. Peak of 34.0 exceeded human level at only 500K training steps.

---

## How to Run

### Install dependencies
```bash
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom opencv-python
AutoROM --accept-license
```

### Train (each member runs their own script)
```bash
python "Victoria_Average Hyperparameters/train.py"
python "Pretty_Lower Hyperparameters/train.py"
python "Erneste_Higher Hyperparameters/train.py"
```

### Play (shared — uses group best model)
```bash
python play.py
```

### Compare MLP vs CNN (shared)
```bash
python compare.py
```

---

## Group Contributions

| Member | Contribution |
|---|---|
| **Victoria Fakunle** | train.py (average hyperparams), compare.py, 10 experiments, repo setup, README |
| **Pretty Ntakirutimana** | train.py (lower hyperparams), 10 experiments including 500K expBEST run |
| **Erneste** | train.py (higher hyperparams), 10 experiments |

---

## References

- Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*. DeepMind.
- Stable Baselines 3 Documentation: https://stable-baselines3.readthedocs.io
- Gymnasium Atari Environments: https://gymnasium.farama.org/environments/atari/
