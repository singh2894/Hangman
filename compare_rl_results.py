# compare_rl_results.py

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(algorithm_name):
    # load training result from saved file
    results_path = f'results/{algorithm_name.lower()}_results.txt'
    
    # Parse results file
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Convert to appropriate type
                try:
                    if '.' in value:
                        results[key] = float(value)
                    else:
                        results[key] = int(value)
                except ValueError:
                    results[key] = value
    
    return results


def load_model(algorithm_name):
    # load Q-table
    model_path = f'models/{algorithm_name.lower()}_model.pkl'
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def print_comparison_table(q_results, s_results):
    # print comparison table
    print("Performance Comparison Table")
    print(f"{'Metric':<30} {'Q-Learning':>15} {'SARSA':>15} {'Difference':>10}")
    
    # Win Rate
    q_wr = q_results['win_rate'] * 100
    s_wr = s_results['win_rate'] * 100
    diff_wr = q_wr - s_wr
    print(f"{'Win Rate (%)':<30} {q_wr:>15.2f} {s_wr:>15.2f} {diff_wr:>+10.2f}")
    
    # Avg Steps
    q_steps = q_results['avg_steps']
    s_steps = s_results['avg_steps']
    diff_steps = q_steps - s_steps
    print(f"{'Avg Steps per Game':<30} {q_steps:>15.2f} {s_steps:>15.2f} {diff_steps:>+10.2f}")
    
    # Avg Reward
    q_reward = q_results['avg_reward']
    s_reward = s_results['avg_reward']
    diff_reward = q_reward - s_reward
    print(f"{'Avg Reward per Game':<30} {q_reward:>15.2f} {s_reward:>15.2f} {diff_reward:>+10.2f}")
    
    # Q-table Size
    q_states = q_results['q-table_size']
    s_states = s_results['q-table_size']
    diff_states = q_states - s_states
    print(f"{'States Explored':<30} {q_states:>15} {s_states:>15} {diff_states:>+10}")

def analyze_performance(q_results, s_results):
    # detail performance analysis
    # 1. Win Rate Analysis
    print("\n1. Win Rate Analysis")
    
    q_wr = q_results['win_rate'] * 100
    s_wr = s_results['win_rate'] * 100
    diff = q_wr - s_wr
    rel_improve = (diff / s_wr) * 100 if s_wr > 0 else 0
    
    print(f"Q-Learning: {q_wr:.2f}%")
    print(f"SARSA:      {s_wr:.2f}%")
    print(f"Absolute difference: {diff:+.2f} percentage points")
    print(f"Relative improvement: {rel_improve:+.1f}%")
    
    if abs(diff) < 1:
        print("Performance is essentially equivalent")
    elif diff > 0:
        print(f"Q-Learning performs better")
    else:
        print(f"SARSA performs better")
    
    # 2. Efficiency Analysis
    print("\n#2. Efficiency Analysis")
    
    q_steps = q_results['avg_steps']
    s_steps = s_results['avg_steps']
    
    print(f"Average steps per game:")
    print(f"  Q-Learning: {q_steps:.2f}")
    print(f"  SARSA:      {s_steps:.2f}")
    
    if q_steps < s_steps:
        improve = ((s_steps - q_steps) / s_steps) * 100
        print(f"Q-Learning is {improve:.1f}% more efficient")
    elif s_steps < q_steps:
        improve = ((q_steps - s_steps) / q_steps) * 100
        print(f"SARSA is {improve:.1f}% more efficient")
    else:
        print(f"Both equally efficient")
    
    # 3. Reward Analysis
    print("\n3. Reward Analysis")
    
    q_reward = q_results['avg_reward']
    s_reward = s_results['avg_reward']
    
    print(f"Average reward per game:")
    print(f"  Q-Learning: {q_reward:.3f}")
    print(f"  SARSA:      {s_reward:.3f}")
    
    if q_reward > s_reward:
        print(f"Q-Learning achieved higher rewards")
    elif s_reward > q_reward:
        print(f"SARSA achieved higher rewards")
    else:
        print(f"Both achieved equal rewards")
    
    # 4. State Space Analysis
    print("\n4. State Space Analysis")
    
    q_states = q_results['q-table_size']
    s_states = s_results['q-table_size']
    
    print(f"Unique states explored:")
    print(f"  Q-Learning: {q_states:,}")
    print(f"  SARSA:      {s_states:,}")
    
    if q_states > s_states:
        diff_pct = ((q_states - s_states) / s_states) * 100
        print(f"Q-Learning explored {diff_pct:.1f}% more states")
        print(f"  (More aggressive exploration)")
    elif s_states > q_states:
        diff_pct = ((s_states - q_states) / q_states) * 100
        print(f"SARSA explored {diff_pct:.1f}% more states")
    else:
        print(f"Both explored same number of states")

def generate_visualizations(q_results, s_results):
    # create visualizations of Q-Learning and SARSA performance comparison
    print("Generating visualizations..")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Win Rate Comparison
    ax = axes[0]
    algorithms = ['Q-Learning', 'SARSA']
    win_rates = [q_results['win_rate'] * 100, s_results['win_rate'] * 100]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax.bar(algorithms, win_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(win_rates) * 1.3])
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
    q_wr_norm = q_results['win_rate'] * 100
    s_wr_norm = s_results['win_rate'] * 100
    
    # Efficiency (inverse of steps, normalized)
    max_steps = max(q_results['avg_steps'], s_results['avg_steps'])
    q_eff = (1 - q_results['avg_steps'] / max_steps) * 100
    s_eff = (1 - s_results['avg_steps'] / max_steps) * 100
    
    # Reward (shift to positive and normalize)
    min_reward = min(q_results['avg_reward'], s_results['avg_reward'])
    max_reward = max(q_results['avg_reward'], s_results['avg_reward'])
    if max_reward != min_reward:
        q_rew_norm = ((q_results['avg_reward'] - min_reward) / (max_reward - min_reward)) * 100
        s_rew_norm = ((s_results['avg_reward'] - min_reward) / (max_reward - min_reward)) * 100
    else:
        q_rew_norm = s_rew_norm = 50
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [q_wr_norm, q_eff, q_rew_norm], width, 
                   label='Q-Learning', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, [s_wr_norm, s_eff, s_rew_norm], width, 
                   label='SARSA', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    
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

    except FileNotFoundError as e:
        print("\nPerformance result file cannot be found.")
        print("\nPlease run training first: python3 train_rl.py")
        return
    
    # print summary
    print(f"\nTraining configuration:")
    print(f"  Episodes: {q_results['episodes']:,}")
    print(f"  Evaluation games: 1,000 (test set)")
    
    # comparison table
    print_comparison_table(q_results, s_results)

    # detail analysis
    analyze_performance(q_results, s_results)
    
    # visualizations
    generate_visualizations(q_results, s_results)
    
    print("Analysis is done. Generate the result to results/rl_comparison.png")


if __name__ == "__main__":
    generate_report()