# compare_rl_results.py

import os
import matplotlib.pyplot as plt
import numpy as np
from dict import words
from heuristic_agent import HeuristicHangmanAgent
from hangman_env import WordSampler, EnvHangman


def load_results(algorithm_name):
    # load training/evaluation result from saved file
    results_path = f'results/{algorithm_name.lower()}_results.txt'

    # parse results file
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip().lower().replace(' ', '_').replace('-', '_')
                value = value.strip()

                # convert to numeric when possible
                try:
                    if '.' in value:
                        results[key] = float(value)
                    else:
                        results[key] = int(value)
                except ValueError:
                    results[key] = value

    return results


def evaluate_heuristic(episodes=1000, seed=999, split="test"):
    # evaluate heuristic agent on held-out words
    sampler = WordSampler(seed=seed)
    agent = HeuristicHangmanAgent(words)

    wins = 0
    total_steps = 0
    total_reward = 0.0

    for ep in range(episodes):
        word = sampler.sample(split=split, seed=seed + ep)
        env = EnvHangman(word)
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False

        while not (terminated or truncated):
            letter = agent.choose_letter(info["pattern"], info["guessed"])
            obs, reward, terminated, truncated, info = env.step(letter)
            total_reward += float(reward)
            total_steps += 1

        if "_" not in info["pattern"]:
            wins += 1

    return {
        "algorithm": "Heuristic",
        "episodes": 0,
        "q_table_size": "N/A",
        "win_rate": wins / episodes,
        "avg_steps": total_steps / episodes,
        "avg_reward": total_reward / episodes,
    }


def save_heuristic_results(results):
    os.makedirs('results', exist_ok=True)
    results_path = 'results/heuristic_results.txt'
    with open(results_path, 'w') as f:
        f.write("Algorithm: Heuristic\n")
        f.write("Episodes: 0\n")
        f.write("Q-table size: N/A\n")
        f.write(f"Win rate: {results['win_rate']:.4f}\n")
        f.write(f"Avg steps: {results['avg_steps']:.4f}\n")
        f.write(f"Avg reward: {results['avg_reward']:.4f}\n")
    print(f"✓ Saved heuristic results: {results_path}")


def load_or_compute_heuristic_results(eval_episodes=1000):
    # try loading existing heuristic results first
    try:
        return load_results("Heuristic")
    except FileNotFoundError:
        print("\nHeuristic result file not found. Evaluating heuristic agent...")
        results = evaluate_heuristic(episodes=eval_episodes, seed=999, split="test")
        save_heuristic_results(results)
        return results


def _safe_metric(results, key, default=np.nan):
    return results.get(key, default)


def print_comparison_table(all_results):
    # print comparison table
    print("Performance Comparison Table")
    print(f"{'Metric':<24} {'Q-Learning':>12} {'SARSA':>12} {'Heuristic':>12}")
    print("-" * 64)

    q_results = all_results["Q-Learning"]
    s_results = all_results["SARSA"]
    h_results = all_results["Heuristic"]

    print(
        f"{'Win Rate (%)':<24} "
        f"{_safe_metric(q_results, 'win_rate') * 100:>12.2f} "
        f"{_safe_metric(s_results, 'win_rate') * 100:>12.2f} "
        f"{_safe_metric(h_results, 'win_rate') * 100:>12.2f}"
    )
    print(
        f"{'Avg Steps per Game':<24} "
        f"{_safe_metric(q_results, 'avg_steps'):>12.2f} "
        f"{_safe_metric(s_results, 'avg_steps'):>12.2f} "
        f"{_safe_metric(h_results, 'avg_steps'):>12.2f}"
    )
    print(
        f"{'Avg Reward per Game':<24} "
        f"{_safe_metric(q_results, 'avg_reward'):>12.2f} "
        f"{_safe_metric(s_results, 'avg_reward'):>12.2f} "
        f"{_safe_metric(h_results, 'avg_reward'):>12.2f}"
    )
    print(
        f"{'States Explored':<24} "
        f"{str(_safe_metric(q_results, 'q_table_size', 'N/A')):>12} "
        f"{str(_safe_metric(s_results, 'q_table_size', 'N/A')):>12} "
        f"{str(_safe_metric(h_results, 'q_table_size', 'N/A')):>12}"
    )


def analyze_performance(all_results):
    # detail performance analysis
    names = ["Q-Learning", "SARSA", "Heuristic"]
    wr = {name: all_results[name]["win_rate"] * 100 for name in names}
    steps = {name: all_results[name]["avg_steps"] for name in names}
    rewards = {name: all_results[name]["avg_reward"] for name in names}

    # 1. Win Rate Analysis
    print("\n1. Win Rate Analysis")
    for name in names:
        print(f"{name:<10}: {wr[name]:.2f}%")
    best_wr = max(wr, key=wr.get)
    print(f"Best win rate: {best_wr} ({wr[best_wr]:.2f}%)")

    # 2. Efficiency Analysis
    print("\n2. Efficiency Analysis")
    for name in names:
        print(f"{name:<10}: {steps[name]:.2f} steps/game")
    best_eff = min(steps, key=steps.get)
    print(f"Most efficient (fewest steps): {best_eff} ({steps[best_eff]:.2f})")

    # 3. Reward Analysis
    print("\n3. Reward Analysis")
    for name in names:
        print(f"{name:<10}: {rewards[name]:.3f}")
    best_rew = max(rewards, key=rewards.get)
    print(f"Highest average reward: {best_rew} ({rewards[best_rew]:.3f})")

    # 4. State Space Analysis (RL only)
    print("\n4. State Space Analysis")
    q_states = all_results["Q-Learning"].get('q_table_size')
    s_states = all_results["SARSA"].get('q_table_size')
    print(f"Q-Learning states: {q_states:,}")
    print(f"SARSA states:      {s_states:,}")
    print("Heuristic states:  N/A (no Q-table)")


def generate_visualizations(all_results):
    # create visualizations of Q-Learning, SARSA, and Heuristic performance comparison
    print("Generating visualizations..")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = ['Q-Learning', 'SARSA', 'Heuristic']
    colors = ['#2ecc71', '#3498db', '#e67e22']

    # Plot 1: Win Rate Comparison
    ax = axes[0]
    win_rates = [all_results[name]['win_rate'] * 100 for name in names]

    bars = ax.bar(names, win_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
    max_wr = max(win_rates) if win_rates else 1
    ax.set_ylim([0, max(10, max_wr * 1.3)])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, val in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: All Metrics Comparison (Normalized)
    ax = axes[1]
    metrics = ['Win Rate', 'Efficiency\n(inv. steps)', 'Reward\n(normalized)']

    # Normalize metrics to 0-100 scale for comparison
    wr_norm = [all_results[name]['win_rate'] * 100 for name in names]

    # Efficiency (inverse of steps, normalized)
    steps_vals = [all_results[name]['avg_steps'] for name in names]
    max_steps = max(steps_vals) if steps_vals else 1
    eff_norm = [(1 - (v / max_steps)) * 100 if max_steps > 0 else 0 for v in steps_vals]

    # Reward (shift to positive and normalize)
    reward_vals = [all_results[name]['avg_reward'] for name in names]
    min_reward = min(reward_vals)
    max_reward = max(reward_vals)
    if max_reward != min_reward:
        rew_norm = [((v - min_reward) / (max_reward - min_reward)) * 100 for v in reward_vals]
    else:
        rew_norm = [50 for _ in reward_vals]

    x = np.arange(len(metrics))
    width = 0.24

    for idx, name in enumerate(names):
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            [wr_norm[idx], eff_norm[idx], rew_norm[idx]],
            width,
            label=name,
            color=colors[idx],
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )

    ax.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # save the visualization
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/rl_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: results/rl_comparison.png")

    plt.show()


def generate_report():
    # main function to generate comparison report
    print("Reinforcement Learning Comparison and Analysis")

    # load results
    try:
        print("\nLoading results...")
        q_results = load_results("Q-Learning")
        s_results = load_results("SARSA")
        eval_episodes = 1000
        h_results = load_or_compute_heuristic_results(eval_episodes=eval_episodes)

    except FileNotFoundError as e:
        print("\nPerformance result file cannot be found.")
        print("\nPlease run training first: python3 train_rl.py")
        return

    all_results = {
        "Q-Learning": q_results,
        "SARSA": s_results,
        "Heuristic": h_results,
    }

    # print summary
    print(f"\nTraining configuration:")
    print(f"  Episodes: {q_results['episodes']:,}")
    print(f"  Evaluation games: 1,000 (test set)")

    # comparison table
    print_comparison_table(all_results)

    # detail analysis
    analyze_performance(all_results)

    # visualizations
    generate_visualizations(all_results)

    print("Analysis is done. Generate the result to results/rl_comparison.png")


if __name__ == "__main__":
    generate_report()
