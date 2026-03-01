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
├── dict.py                    # 500 five-letter words
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

Performance after 50,000 training episodes:

| Algorithm | Win Rate | Avg Reward | States Explored |
|-----------|----------|------------|-----------------|
| **Q-Learning** | 18.6% | -1.14 | 218,279 |
| **SARSA** | 14.7% | -1.81 | 177,341 |
| **Heuristic** | 95%+ | +5.2 | N/A |

**Key Findings:**
- Q-Learning outperforms SARSA by 3.9 percentage points (26.5% relative improvement)
- Off-policy learning explores 23% more states
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
alpha | Q-Learning    | SARSA
------|---------------|-------------
0.01  | 34.3% ± 3.6% | 34.5% ± 4.4%
0.05  | 36.0% ± 8.5% | 30.3% ± 3.4% ⭐
0.10  | 24.7% ± 6.9% | 16.1% ± 7.9%
0.20  |  9.6% ± 2.1% |  9.3% ± 2.9%
0.30  |  4.1% ± 1.3% |  3.1% ± 1.2%

Optimal: α=0.05 (Q-Learning), α=0.01 (SARSA)
```

---

## Configuration

**Training:**
- Episodes: 50,000
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
| `dict.py` | 500 five-letter word dictionary |
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
Metric                    Q-Learning         SARSA    Difference
Win Rate (%)                   18.60         14.70        +3.90
Avg Steps per Game              8.60          8.34        +0.26
Avg Reward per Game            -1.14         -1.81        +0.67
States Explored              218,279       177,341      +40,938
```

---

## Contributing

Contributions welcome! Please submit a Pull Request.

---

## License

MIT License - see LICENSE file for details.

---

## References

- Sutton & Barto: *Reinforcement Learning: An Introduction*
- Gymnasium: https://gymnasium.farama.org/

---

**Made with ❤️ for RL learning**
