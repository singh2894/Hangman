# main.py
from dict import words
from heuristic_agent import HeuristicHangmanAgent
from hangman_env import WordSampler, EnvHangmanGym
import argparse
import time

#ASCII Hangman drawing
HANGMAN_STAGES = [
    # Stage 0
    """
       +---+
       |   |
           |
           |
           |
           |
    =========
    """,
    # Stage 1
    """
       +---+
       |   |
       O   |
           |
           |
           |
    =========
    """,
    # Stage 2
    """
       +---+
       |   |
       O   |
       |   |
           |
           |
    =========
    """,
    # Stage 3
    """
       +---+
       |   |
       O   |
      /|   |
           |
           |
    =========
    """,
    # Stage 4
    """
       +---+
       |   |
       O   |
      /|\\  |
           |
           |
    =========
    """,
    # Stage 5
    """
       +---+
       |   |
       O   |
      /|\\  |
      /    |
           |
    =========
    """,
    # Stage 6
    """
       +---+
       |   |
       O   |
      /|\\  |
      / \\  |
           |
    =========
    GAME OVER!
    """
]

def letter_to_action(letter: str) -> int:
    return ord(letter.lower()) - ord("a")


def run_gym(seed=42, split="train"):
    sampler = WordSampler(seed=seed)
    secret_word = sampler.sample(split=split, seed=None)

    env = EnvHangmanGym(secret_word)
    agent = HeuristicHangmanAgent(words)
    obs, info = env.reset(seed=seed)

    print(f"Secret word: {info['word']}")
    print("-" * 40)

    # Show initial state
    print("\n" + "=" * 60)
    print("INITIAL STATE:")
    print(f"Pattern: {' '.join(info['pattern'])}")
    print(HANGMAN_STAGES[0])  # Show empty gallows
    print("=" * 60)

    done = False
    while not done:
        letter = agent.choose_letter(info["pattern"], info["guessed"])
        action = letter_to_action(letter)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        mistakes_cnt = 6 - info['wrong_left']

        print(f"Guessed  : {letter}")
        print(f"Pattern  : {' '.join(info['pattern'])}")
        print(f"Guessed  : {info['guessed']}")
        print(f"Wrong left: {info['wrong_left']}  |  Reward: {reward}")
        print()
        print(HANGMAN_STAGES[mistakes_cnt])
        print("-" * 60)

        # Add delay to print the ASCII Hangman drawing
        time.sleep(0.5) 

    if "_" not in info["pattern"]:
        print("Agent WON 🎉")
    else:
        print("Agent LOST ❌")

    print(f"Final word: {info['word']}")


def run():
    parser = argparse.ArgumentParser(description="Run heuristic Hangman agent.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()

    run_gym(seed=args.seed, split=args.split)



if __name__ == "__main__":
    run()