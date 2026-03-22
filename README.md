# DQN Breakout — Group Hyperparameter Experiment
**Formative 3 · Deep Q-Learning · Stable Baselines 3 + Gymnasium**

---

## Group Members

| Member | Role | Hyperparameter Range |
|---|---|---|
| **Victoria Fakunle** | Average Hyperparameters | lr=0.0001–0.0003, γ=0.97–0.99, batch=32–64 |
| **Pretty** | Lower Hyperparameters | *(fill in range)* |
| **Erneste** | Higher Hyperparameters | lr=0.0005–0.001, γ=0.99–0.999, batch=128–256 |

---

## Environment

| Field | Value |
|---|---|
| Game | ALE/Breakout-v5 |
| Framework | Stable Baselines 3 |
| Policy | CNNPolicy (Convolutional Neural Network) |
| Observation | 84×84 grayscale · 4-frame stack |
| Training steps | 500,000 per experiment |
| Total experiments | 30 (10 per member) |

---

## Gameplay Video

> 📹 **[INSERT VIDEO LINK HERE]**
> *(Recorded using play.py with the best model across all 30 experiments)*

---

## Repository Structure

```
├── Best_Model/
│   └── dqn_model.zip                        # Best model across all 30 experiments
│
├── Victoria_Average Hyperparameters/
│   ├── train.py                             # Victoria's 10 experiments
│   ├── models/
│   └── Logs/
│
├── Pretty_Lower Hyperparameters/
│   ├── train.py                             # Pretty's 10 experiments
│   ├── models/
│   └── Logs/
│
├── Erneste_Higher Hyperparameters/
│   ├── train.py                             # Erneste's 10 experiments
│   ├── models/
│   └── Logs/
│
├── play.py                                  # Loads best model
├── compare.py                               # Shared: MLP vs CNN
└── README.md
```

---

## Policy Comparison — MLP vs CNN

CNNPolicy is the correct architecture for pixel-based Atari environments. The input is a stack of 4 grayscale frames (84×84×4), which requires spatial feature extraction. CNNPolicy applies convolutional layers to detect edges, shapes, and motion before making Q-value decisions. MlpPolicy flattens the entire image into a 1D vector, destroying all spatial structure and making it nearly impossible to learn from raw pixels.

Run `compare.py` to reproduce these results.

| Policy | Mean Reward | Std | Notes |
|---|---|---|---|
| CnnPolicy | 13.20 | 5.00 | Standard for pixel-based RL |
| MlpPolicy | 4.60 |1.74 | Flattens pixels — loses spatial structure |

**Both policies were trained for 100,000 steps using the same hyperparameters (lr=0.0001, gamma=0.98, batch=64). CnnPolicy scored 13.20 vs MlpPolicy's 4.60 — a difference of +8.60. MlpPolicy peaked at 25K steps (6.60) then declined to 4.60, confirming it cannot learn effectively from raw pixels. CNNPolicy improved consistently across all 4 checkpoints, reaching 13.90 at 75K before settling at 13.20.**
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

### Pretty — Lower Hyperparameters


| Experiment | Timesteps | Final Mean Reward | Peak Mean Reward | Noted Behaviour |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 01** | 100,000 | 2.4 | 2.4 | Extremely slow learning; minimal improvement throughout the run. [cite: 10] |
| **Exp 02** | 100,000 | 2.6 | 5.0 | Initial promise at 40k, but performance dropped significantly thereafter. [cite: 11] |
| **Exp 03** | 100,000 | 5.4 | 6.2 | Strong initial start (6.2 at 20k) but highly unstable, dropping as low as 1.0. [cite: 12] |
| **Exp 04** | 100,000 | 8.4 | 8.4 | Consistent but slow improvement; reached its peak at the very end. [cite: 13] |
| **Exp 05** | 100,000 | 7.0 | 11.2 | Reached a high peak at 80k but suffered from performance degradation in the final 20k steps. [cite: 14] |
| **Exp 06** | 100,000 | 12.6 | 12.6 | Strong final performance, though it experienced a notable dip in rewards around 80k. [cite: 15] |
| **Exp 07** | 100,000 | 4.0 | 12.6 | Highest potential mid-run (12.6 at 60k) but suffered a catastrophic collapse in reward by 100k. [cite: 16] |
| **Exp 08** | 100,000 | 10.4 | 10.4 | Demonstrates very steady and stable growth throughout the training session. [cite: 17] |
| **Exp 09 (BEST)** | 500,000 | 23.8 | 23.8 | Outstanding performance; sustained learning over a longer duration, achieving the highest reward. [cite: 19] |
| **Exp 10** | 100,000 | 8.8 | 8.8 | Modest but consistent growth, showing similar stability to Exp 08. [cite: 18] |


**Pretty's best:** exp09 — Mean Reward **23.8**

---

### Erneste — Higher Hyperparameters

| # | lr | γ | batch | eps_start | eps_end | eps_frac | Mean Reward | Std | Peak | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| exp01 | 0.0005 | 0.990 | 128 | 1.0 | 0.10 | 0.15 | 26.70 | 8.16 | 26.70 | Reward improving across training. |
| exp02 | 0.0005 | 0.995 | 128 | 1.0 | 0.10 | 0.20 | 23.85 | 8.90 | 24.80 | Reward improving across training. |
| exp03 | 0.0005 | 0.999 | 256 | 1.0 | 0.10 | 0.20 | 23.45 | 7.52 | 23.65 | Reward flat — may need more timesteps. |
| exp04 | 0.0007 | 0.990 | 128 | 1.0 | 0.10 | 0.15 | 23.45 | 6.74 | 29.25 | Reward declined — possible instability. |
| exp05 | 0.0007 | 0.995 | 256 | 1.0 | 0.10 | 0.20 | 23.25 | 6.68 | 24.75 | Reward flat — may need more timesteps. |
| exp06 | 0.0007 | 0.999 | 256 | 1.0 | 0.15 | 0.25 | 23.70 | 7.41 | 24.45 | Reward improving across training. |
| exp07 | 0.0010 | 0.990 | 128 | 1.0 | 0.10 | 0.15 | 27.05 | 5.54 | 27.05 | Reward improving across training. |
| exp08 | 0.0010 | 0.995 | 256 | 1.0 | 0.15 | 0.25 | **27.80** | 7.51 | 30.90 | **Best for Erneste — high batch size helped stability** |
| exp09 | 0.0010 | 0.999 | 256 | 1.0 | 0.15 | 0.30 | 18.30 | 6.72 | 26.30 | Reward declined — possible instability. |
| exp10 | 0.0007 | 0.999 | 128 | 1.0 | 0.10 | 0.25 | 21.30 | 8.20 | 28.25 | Reward declined — possible instability. |

**Erneste's best:** exp08 — Mean Reward **27.80**

---

## Best Model Selection

| Member | Best Exp | Mean Reward | Model Location |
|---|---|---|---|
| Victoria Fakunle | exp02 | 31.80 | Victoria_Average Hyperparameters/models/ |
| Pretty Ntakirutiman | exp09 | 23.8 | Pretty_Lower Hyperparameters/models/ |
| Erneste | exp08 | 27.80 | Erneste_Higher Hyperparameters/models/ |
| **Group best** | **Victoria exp02** | **31.80** | **Best_Model/dqn_model.zip** |

> The winning model is copied to `Best_Model/dqn_model.zip` and used by `play.py`.

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
- RAM pressure during exp08's original run caused buffer swapping, which degraded training quality.

**Best config:** `lr=0.0001, γ=0.98, batch=64, eps_end=0.05, eps_frac=0.10` → Mean Reward **31.80**

---

### Pretty — Lower Hyperparameters

> *(Fill in: which hyperparams improved performance, which harmed it, best config and why)*

---

### Erneste — Higher Hyperparameters

**What improved performance:**
- `batch=256` significantly improved gradient stability compared to smaller batch sizes in the average sweep.
- Higher learning rate (`lr=0.0010`) combined with larger batches allowed faster convergence without overshooting.
- `γ=0.995` provided a better balance for long-term planning than γ=0.990 in this higher range.

**What harmed performance:**
- `γ=0.999` remains difficult to train at 500K steps, often leading to performance dips or flat rewards.
- Conservative exploration (`eps_frac=0.30`) slowed down the policy's ability to capitalize on learned strategies.
- Instability was observed at high learning rates if the batch size was not sufficiently large.

**Best config:** `lr=0.0010, γ=0.995, batch=256, eps_end=0.15, eps_frac=0.25` → Mean Reward **27.80** (Peak 30.90)

---

## How to Run

### Install dependencies
```bash
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom opencv-python
AutoROM --accept-license
```

### Train (each member runs their own script)
```bash
# Victoria
python "Victoria_Average Hyperparameters/train.py"

# Pretty
python "Pretty_Lower Hyperparameters/train.py"

# Erneste
python "Erneste_Higher Hyperparameters/train.py"
```

### Play (shared — uses best model from all 30 experiments)
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
| **Pretty** | train.py (lower hyperparams), 10 experiments *(add detail)* |
| **Erneste** | train.py (higher hyperparams), 10 experiments, play.py, README |

---

## References

- Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*. DeepMind.
- Stable Baselines 3 Documentation: https://stable-baselines3.readthedocs.io
- Gymnasium Atari Environments: https://gymnasium.farama.org/environments/atari/
