# Hangman Reinforcement Learning

Implementation and comparison of Q-Learning and SARSA algorithms for solving the Hangman word-guessing game.

## Overview

Train RL agents to play Hangman and compare their performance against a heuristic baseline.

**Key Features:**
- Custom Gymnasium environment for 5-letter Hangman
- Q-Learning and SARSA implementations
- Pattern-matching heuristic agent
- Hyperparameter tuning with grid search
- Performance visualization and analysis

---

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/hangman-rl.git
cd hangman-rl

# Install dependencies
pip install numpy gymnasium matplotlib
```

---

## Quick Start
```bash
# 1. Watch heuristic agent play (with ASCII visualization)
python main.py

# 2. Train RL agents (~20-30 minutes)
python train_rl.py

# 3. Compare results and generate visualizations
python compare_rl_results.py
```

---

## Project Structure
```
hangman-rl/
├── dict.py                   # 1,000 five-letter words
├── hangman_env.py            # Gymnasium environment
├── heuristic_agent.py        # Pattern-matching baseline
├── train_rl.py               # RL training & tuning
├── compare_rl_results.py     # Performance analysis
├── main.py                   # Interactive demo
├── models/                   # Saved Q-tables (generated)
└── results/                  # Metrics & plots (generated)
```

---

## Usage

### Interactive Demo
```bash
python main.py                # Watch agent play
python main.py --seed 42      # Use specific seed
```

### Training
```bash
python train_rl.py                              # Default (50k episodes)
python train_rl.py --episodes 10000             # Custom episodes
python train_rl.py --grid                       # Hyperparameter tuning
```

### Analysis
```bash
python compare_rl_results.py                   # Generate comparison report
```

---

## Results

Performance after 100,000 training episodes:

| Algorithm | Win Rate | Avg Reward | States Explored |
|-----------|----------|------------|-----------------|
| **Q-Learning** | 65.1% | 3.74 | 382,975 |
| **SARSA** | 64.2% | 3.67 | 297,554 |
| **Heuristic** | 98.1%+ | 7.01 | N/A |

**Key Findings:**
- Q-Learning outperforms SARSA by 0.9 percentage points (1.4% relative improvement)
- Off-policy learning explores 28% more states
- Heuristic baseline significantly outperforms RL (uses domain knowledge)

---

## MDP Formulation

**State:** `(pattern, guessed_letters, wrong_left)`  
**Actions:** 26 letters (unguessed only)  
**Rewards:** +1 correct, -1 wrong, +5 win bonus  
**Transition:** Deterministic  

Example:
```
State: "____E", guessed={E}, wrong=6
Action: Guess 'A'
Next State: "A___E", guessed={A,E}, wrong=6
Reward: +1
```

---

## Algorithms

### Q-Learning (Off-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```
- Learns optimal policy while exploring
- Better for deterministic environments

### SARSA (On-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
```
- Learns actual behavior policy
- More conservative, safer learning

---

## Hyperparameter Tuning

Grid search over learning rates with multiple seeds:
```bash
python train_rl.py --grid
```

Results:
```
=== Alpha grid search (win rate on test) ===
Train episodes: 50000 | Eval episodes: 1000 | Seeds: [0, 1, 2, 3, 4]
gamma | eps_end | alpha | Q-learning mean±std | SARSA mean±std
------|---------|-------|---------------------|----------------
0.90 |   0.050 |  0.01 |  0.251±0.040  |  0.204±0.043 
0.90 |   0.050 |  0.05 |  0.152±0.027  |  0.139±0.091 
0.90 |   0.050 |  0.10 |  0.092±0.047  |  0.067±0.030 
0.90 |   0.050 |  0.20 |  0.063±0.034  |  0.029±0.015 
0.90 |   0.050 |  0.30 |  0.044±0.011  |  0.038±0.019 
0.90 |   0.020 |  0.01 |  0.302±0.015  |  0.215±0.059 
0.90 |   0.020 |  0.05 |  0.230±0.036  |  0.191±0.064 
0.90 |   0.020 |  0.10 |  0.132±0.036  |  0.129±0.062 
0.90 |   0.020 |  0.20 |  0.052±0.025  |  0.065±0.012 
0.90 |   0.020 |  0.30 |  0.035±0.012  |  0.029±0.021 
0.90 |   0.010 |  0.01 |  0.322±0.052  |  0.288±0.033 
0.90 |   0.010 |  0.05 |  0.269±0.042  |  0.223±0.053 
0.90 |   0.010 |  0.10 |  0.158±0.069  |  0.123±0.055 
0.90 |   0.010 |  0.20 |  0.067±0.031  |  0.086±0.027 
0.90 |   0.010 |  0.30 |  0.045±0.020  |  0.053±0.022 
0.95 |   0.050 |  0.01 |  0.226±0.038  |  0.244±0.057 
0.95 |   0.050 |  0.05 |  0.188±0.058  |  0.158±0.064 
0.95 |   0.050 |  0.10 |  0.118±0.014  |  0.108±0.036 
0.95 |   0.050 |  0.20 |  0.062±0.012  |  0.044±0.023 
0.95 |   0.050 |  0.30 |  0.030±0.012  |  0.023±0.016 
0.95 |   0.020 |  0.01 |  0.294±0.032  |  0.237±0.042 
0.95 |   0.020 |  0.05 |  0.274±0.030  |  0.200±0.046 
0.95 |   0.020 |  0.10 |  0.161±0.038  |  0.145±0.051 
0.95 |   0.020 |  0.20 |  0.058±0.021  |  0.088±0.028 
0.95 |   0.020 |  0.30 |  0.044±0.010  |  0.063±0.016 
0.95 |   0.010 |  0.01 |  0.306±0.057  |  0.286±0.017 
0.95 |   0.010 |  0.05 |  0.274±0.048  |  0.171±0.046 
0.95 |   0.010 |  0.10 |  0.164±0.040  |  0.161±0.060 
0.95 |   0.010 |  0.20 |  0.106±0.019  |  0.086±0.030 
0.95 |   0.010 |  0.30 |  0.057±0.013  |  0.055±0.033 
0.99 |   0.050 |  0.01 |  0.232±0.034  |  0.240±0.021 
0.99 |   0.050 |  0.05 |  0.163±0.032  |  0.160±0.026 
0.99 |   0.050 |  0.10 |  0.132±0.015  |  0.069±0.018 
0.99 |   0.050 |  0.20 |  0.054±0.028  |  0.034±0.016 
0.99 |   0.050 |  0.30 |  0.043±0.013  |  0.034±0.028 
0.99 |   0.020 |  0.01 |  0.264±0.069  |  0.294±0.024 
0.99 |   0.020 |  0.05 |  0.304±0.072  |  0.240±0.018 
0.99 |   0.020 |  0.10 |  0.151±0.033  |  0.146±0.056 
0.99 |   0.020 |  0.20 |  0.102±0.068  |  0.068±0.023 
0.99 |   0.020 |  0.30 |  0.035±0.007  |  0.023±0.018 
0.99 |   0.010 |  0.01 |  0.303±0.042  |  0.306±0.027 
0.99 |   0.010 |  0.05 |  0.273±0.023  |  0.238±0.037 
0.99 |   0.010 |  0.10 |  0.173±0.050  |  0.165±0.021 
0.99 |   0.010 |  0.20 |  0.087±0.014  |  0.074±0.031 
0.99 |   0.010 |  0.30 |  0.044±0.019  |  0.050±0.026 

Best Q-learning config (gamma, eps_end, alpha): (0.9, 0.01, 0.01) (mean win rate 0.322)
Best SARSA config (gamma, eps_end, alpha): (0.99, 0.01, 0.01) (mean win rate 0.306)
```

---

## Configuration

**Training:**
- Episodes: 100,000
- Learning rate (α): 0.1
- Discount (γ): 0.95
- Exploration: ε = 1.0 → 0.05 (decays over 30k episodes)

**Evaluation:**
- Test episodes: 1,000
- Greedy policy (no exploration)
- Held-out test set (20% of words)

---

## File Descriptions

| File | Purpose |
|------|---------|
| `dict.py` | 1,000 five-letter word dictionary |
| `hangman_env.py` | Gymnasium environment implementation |
| `heuristic_agent.py` | Pattern-matching baseline agent |
| `train_rl.py` | Q-Learning & SARSA training |
| `compare_rl_results.py` | Performance analysis & visualization |
| `main.py` | Interactive demo with ASCII graphics |

---

## Requirements

- Python 3.8+
- numpy >= 1.21.0
- gymnasium >= 0.29.0
- matplotlib >= 3.5.0

Install all: `pip install numpy gymnasium matplotlib`

---

## Example Output

**Training:**
```
=== Q-learning ===
Episode 500/50000 done | Q states: 2544
Episode 1000/50000 done | Q states: 4870
...
Greedy policy win rate: 0.186
✓ Model saved to models/q-learning_model.pkl
```

**Analysis:**
```
Performance Comparison Table
Metric                    Q-Learning         SARSA    Heuristic
Win Rate (%)                   65.10         64.20        98.10
Avg Steps per Game              7.68          7.67         7.29
Avg Reward per Game             3.74          3.67         7.01
States Explored              382,975       297,554          N/A
```
