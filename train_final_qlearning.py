"""
Final training script for the Hangman Q-learning agent.

This script trains the final selected model using the best
hyperparameters identified during grid search.

Best configuration (from experiments):
    alpha   = 0.01
    gamma   = 0.90
    eps_end = 0.01
"""

from train_rl import train_qlearning, evaluate_greedy, evaluate_greedy_return_winrate
from hangman_env import EnvHangmanGym, WordSampler
import numpy as np


if __name__ == "__main__":

    episodes = 500000

    print("\n=== FINAL Q-LEARNING TRAINING ===")
    print("Episodes:", episodes)
    print("alpha = 0.01")
    print("gamma = 0.90")
    print("eps_end = 0.01")

    # Train the model using the tuned hyperparameters
    Q = train_qlearning(
        episodes=episodes,
        alpha=0.01,
        gamma=0.90,
        eps_end=0.01
    )

    # Evaluate the trained policy
    print("\nEvaluating final greedy policy...")

    # Number of explored states in the learned Q-table
    states_explored = len(Q)
    print(f"States explored during training: {states_explored}")

    # Additional evaluation analytics
    eval_results = evaluate_greedy(Q, episodes=1000, split="test")
    print(f"Average steps per game: {eval_results['avg_steps']:.2f}")
    print(f"Average reward per game: {eval_results['avg_reward']:.3f}")

    print("\nTraining complete. Model saved automatically by train_qlearning().")
