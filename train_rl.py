# train_rl.py
import random
import numpy as np
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

    print("Greedy policy win rate:", wins / episodes)
    print("Avg steps per episode:", total_steps / episodes)
    print("Avg reward per episode:", total_reward / episodes)


if __name__ == "__main__":
    EPISODES = 50000
    EVAL_EPISODES = 1000

    print("=== Q-learning ===")
    Q_q = train_qlearning(episodes=EPISODES, split="train")
    evaluate_greedy(Q_q, episodes=EVAL_EPISODES, split="test")

    print("\n=== SARSA ===")
    Q_s = train_sarsa(episodes=EPISODES, split="train")
    evaluate_greedy(Q_s, episodes=EVAL_EPISODES, split="test")