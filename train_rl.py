# train_rl.py
import random
import numpy as np
import os
import pickle
import math
import argparse
from collections import defaultdict

from hangman_env import WordSampler, EnvHangmanGym


def obs_to_key(obs: np.ndarray) -> tuple:
    """Convert numpy observation to a hashable key for tabular Q."""
    return tuple(int(x) for x in obs.tolist())


def letter_to_action(ch: str) -> int:
    """'a'->0, ..., 'z'->25"""
    return ord(ch) - ord("a")


def legal_action_indices(env: EnvHangmanGym) -> list[int]:
    """Only allow letters not guessed yet (legal actions only)."""
    return [letter_to_action(ch) for ch in env._get_legal_actions()]


def epsilon_greedy(Q, s_key, legal, eps):
    """Choose random legal action w.p. eps, else best legal action."""
    if random.random() < eps:
        return random.choice(legal)
    return max(legal, key=lambda a: float(Q[s_key][a]))


def max_over_legal(Q, s_key, legal):
    """Max Q(s,a) over legal actions only."""
    if not legal:
        return 0.0
    return max(float(Q[s_key][a]) for a in legal)


def train_qlearning(
    episodes=5000,
    alpha=0.1,
    gamma=0.95,
    seed=42,
    split="train",
):
    """Tabular Q-learning training loop."""
    random.seed(seed)
    np.random.seed(seed)

    sampler = WordSampler(seed=seed)
    Q = defaultdict(lambda: np.zeros(26, dtype=np.float32))
    eps_start = 1.0
    eps_end = 0.05
    decay_episodes = int(episodes * 0.6)

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(
            0, (decay_episodes - ep)
        ) / max(1, decay_episodes)
        word = sampler.sample(split=split, seed=seed + ep)
        env = EnvHangmanGym(word)

        obs, info = env.reset(seed=seed + ep)
        s = obs_to_key(obs)

        terminated = truncated = False
        while not (terminated or truncated):
            legal = legal_action_indices(env)
            a = epsilon_greedy(Q, s, legal, eps)

            next_obs, r, terminated, truncated, info = env.step(a)
            s2 = obs_to_key(next_obs)

            next_legal = legal_action_indices(env)
            target = float(r) + gamma * max_over_legal(Q, s2, next_legal)
            Q[s][a] += alpha * (target - float(Q[s][a]))

            s = s2


        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1}/{episodes} done | Q states: {len(Q)}")

    return Q


def train_sarsa(
    episodes=5000,
    alpha=0.1,
    gamma=0.95,
    seed=42,
    split="train",
):
    """Tabular SARSA training loop (on-policy)."""
    random.seed(seed)
    np.random.seed(seed)

    sampler = WordSampler(seed=seed)
    Q = defaultdict(lambda: np.zeros(26, dtype=np.float32))

    eps_start = 1.0
    eps_end = 0.05
    decay_episodes = int(episodes * 0.6)

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0, (decay_episodes - ep)) / max(1, decay_episodes)

        word = sampler.sample(split=split, seed=seed + ep)
        env = EnvHangmanGym(word)

        obs, info = env.reset(seed=seed + ep)
        s = obs_to_key(obs)

        # choose initial action
        legal = legal_action_indices(env)
        a = epsilon_greedy(Q, s, legal, eps)

        terminated = truncated = False
        while not (terminated or truncated):
            next_obs, r, terminated, truncated, info = env.step(a)
            s2 = obs_to_key(next_obs)

            if terminated or truncated:
                target = float(r)
                Q[s][a] += alpha * (target - float(Q[s][a]))
                break

            next_legal = legal_action_indices(env)
            a2 = epsilon_greedy(Q, s2, next_legal, eps)

            target = float(r) + gamma * float(Q[s2][a2])
            Q[s][a] += alpha * (target - float(Q[s][a]))

            s, a = s2, a2

        if (ep + 1) % 500 == 0:
            print(f"[SARSA] Episode {ep+1}/{episodes} done | Q states: {len(Q)}")

    return Q


def evaluate_greedy(Q, episodes=200, seed=999, split="test"):
    """Evaluate greedy policy derived from Q on held-out words."""
    sampler = WordSampler(seed=seed)
    wins = 0
    total_steps = 0
    total_reward = 0.0

    for ep in range(episodes):
        word = sampler.sample(split=split, seed=seed + ep)
        env = EnvHangmanGym(word)

        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False

        while not (terminated or truncated):
            s = obs_to_key(obs)
            legal = legal_action_indices(env)
            a = max(legal, key=lambda x: float(Q[s][x]))
            obs, r, terminated, truncated, info = env.step(a)
            total_reward += float(r)
            total_steps += 1

        if "_" not in info["pattern"]:
            wins += 1

    # calculate metrics
    win_rate = wins / episodes
    avg_steps = total_steps / episodes
    avg_reward = total_reward / episodes

    print("Greedy policy win rate:", win_rate)
    print("Avg steps per episode:", avg_steps)
    print("Avg reward per episode:", avg_reward)

    return {
        'win_rate': win_rate,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward
    }


def evaluate_greedy_return_winrate(Q, episodes=1000, seed=999, split="test") -> float:
    """Evaluate greedy policy derived from Q and return win rate as a float."""
    sampler = WordSampler(seed=seed)
    wins = 0

    for ep in range(episodes):
        word = sampler.sample(split=split, seed=seed + ep)
        env = EnvHangmanGym(word)

        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False

        while not (terminated or truncated):
            s = obs_to_key(obs)
            legal = legal_action_indices(env)
            a = max(legal, key=lambda x: float(Q[s][x]))
            obs, r, terminated, truncated, info = env.step(a)

        if "_" not in info["pattern"]:
            wins += 1

    return wins / episodes


def mean_std(xs):
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(v)


def grid_search_alpha(
    alphas=(0.01, 0.05, 0.1, 0.2, 0.3),
    train_episodes=50000,
    eval_episodes=1000,
    seeds=(0, 1, 2, 3, 4),
    split_train="train",
    split_eval="test",
):
    """Professor-style alpha sweep.

    - Fix everything except alpha.
    - For each alpha, train with multiple seeds.
    - Evaluate greedy win rate on held-out split.
    - Print mean ± std for Q-learning and SARSA.

    Note: This retrains for each (alpha, seed, algorithm) combination.
    """

    results = []

    for alpha in alphas:
        q_wins = []
        s_wins = []

        for s in seeds:
            # Q-learning
            Q_q = train_qlearning(
                episodes=train_episodes,
                alpha=alpha,
                seed=42 + s,
                split=split_train,
            )
            wr_q = evaluate_greedy_return_winrate(
                Q_q,
                episodes=eval_episodes,
                seed=999 + 1000 * s,
                split=split_eval,
            )
            q_wins.append(wr_q)

            # SARSA
            Q_s = train_sarsa(
                episodes=train_episodes,
                alpha=alpha,
                seed=42 + s,
                split=split_train,
            )
            wr_s = evaluate_greedy_return_winrate(
                Q_s,
                episodes=eval_episodes,
                seed=1999 + 1000 * s,
                split=split_eval,
            )
            s_wins.append(wr_s)

        q_mean, q_std = mean_std(q_wins)
        s_mean, s_std = mean_std(s_wins)

        results.append((alpha, q_mean, q_std, s_mean, s_std))

    print("\n=== Alpha grid search (win rate on test) ===")
    print(f"Train episodes: {train_episodes} | Eval episodes: {eval_episodes} | Seeds: {list(seeds)}")
    print("alpha | Q-learning mean±std | SARSA mean±std")
    print("------|---------------------|----------------")
    for alpha, q_mean, q_std, s_mean, s_std in results:
        print(f"{alpha:>4.2f} | {q_mean:>6.3f}±{q_std:<6.3f} | {s_mean:>6.3f}±{s_std:<6.3f}")

    best_q = max(results, key=lambda row: row[1])
    best_s = max(results, key=lambda row: row[3])
    print("\nBest Q-learning alpha:", best_q[0], f"(mean win rate {best_q[1]:.3f})")
    print("Best SARSA alpha:", best_s[0], f"(mean win rate {best_s[3]:.3f})")

    return results


def save_results(Q, eval_results, algorithm_name, episodes):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # save Q-table
    model_path = f'models/{algorithm_name.lower()}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(dict(Q), f)
    
    print(f"Model is successfully saved to {model_path}")

    # save result to file
    results_path = f'results/{algorithm_name.lower()}_results.txt'
    with open(results_path, 'w') as f:
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Q-table size: {len(Q)}\n")
        f.write(f"Win rate: {eval_results['win_rate']:.4f}\n")
        f.write(f"Avg steps: {eval_results['avg_steps']:.4f}\n")
        f.write(f"Avg reward: {eval_results['avg_reward']:.4f}\n")

    print(f"Result is successfully saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50000, help="Training episodes")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Evaluation episodes")
    parser.add_argument("--grid", action="store_true", help="Run professor-style alpha grid search")
    args = parser.parse_args()

    if args.grid:
        # Professor-style sweep (edit these lists as you like)
        grid_search_alpha(
            alphas=(0.01, 0.05, 0.1, 0.2, 0.3),
            train_episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seeds=(0, 1, 2, 3, 4),
        )
        raise SystemExit(0)

    EPISODES = args.episodes
    EVAL_EPISODES = args.eval_episodes

    print("=== Q-learning ===")
    Q_q = train_qlearning(episodes=EPISODES, split="train")
    eval_result_q_learning = evaluate_greedy(Q_q, episodes=EVAL_EPISODES, split="test")
    save_results(Q_q, eval_result_q_learning, "Q-Learning", EPISODES)

    print("\n=== SARSA ===")
    Q_s = train_sarsa(episodes=EPISODES, split="train")
    eval_result_sarsa = evaluate_greedy(Q_s, episodes=EVAL_EPISODES, split="test")
    save_results(Q_s, eval_result_sarsa, "SARSA", EPISODES)