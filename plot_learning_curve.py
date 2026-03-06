import matplotlib.pyplot as plt
from train_rl import train_qlearning, train_sarsa, evaluate_greedy_return_winrate
import numpy as np

episodes_per_step = 5000
total_episodes = 50000
steps = total_episodes // episodes_per_step
seeds = [0,1,2,3,4]

q_curve = []
s_curve = []
q_std = []
s_std = []
x = []

# Best tuned hyperparameters from grid search
# Q-learning: gamma=0.90, eps_end=0.01, alpha=0.01
# SARSA: gamma=0.99, eps_end=0.02, alpha=0.01

for i in range(steps):
    ep = (i + 1) * episodes_per_step
    print(f"Training models with {ep} episodes...")

    q_results = []
    s_results = []

    for seed in seeds:
        Qq = train_qlearning(
            episodes=ep,
            alpha=0.01,
            gamma=0.90,
            eps_end=0.01,
            seed=seed,
        )
        Qs = train_sarsa(
            episodes=ep,
            alpha=0.01,
            gamma=0.99,
            eps_end=0.02,
            seed=seed,
        )

        wr_q = evaluate_greedy_return_winrate(Qq, episodes=1000)
        wr_s = evaluate_greedy_return_winrate(Qs, episodes=1000)

        q_results.append(wr_q)
        s_results.append(wr_s)

    q_curve.append(np.mean(q_results))
    s_curve.append(np.mean(s_results))
    q_std.append(np.std(q_results))
    s_std.append(np.std(s_results))
    x.append(ep)

    print(f"Episodes {ep}: Q-learning win rate={np.mean(q_results):.3f}, SARSA win rate={np.mean(s_results):.3f}")

plt.plot(x, q_curve, marker="o", label="Q-learning")
plt.plot(x, s_curve, marker="o", label="SARSA")

plt.fill_between(x, np.array(q_curve)-np.array(q_std), np.array(q_curve)+np.array(q_std), alpha=0.2)
plt.fill_between(x, np.array(s_curve)-np.array(s_std), np.array(s_curve)+np.array(s_std), alpha=0.2)

plt.xlabel("Training Episodes")
plt.ylabel("Win Rate")
plt.title("Learning Curve: Tuned Hangman RL Agents")
plt.ylim(0, 0.4)
plt.legend()
plt.grid(True)

plt.savefig("results/learning_curve.png", dpi=300)
plt.show()