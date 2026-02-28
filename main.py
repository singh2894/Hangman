# main.py
from dict import words
from heuristic_agent import HeuristicHangmanAgent
from hangman_env import WordSampler, EnvHangmanGym
import argparse


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

    done = False
    while not done:
        letter = agent.choose_letter(info["pattern"], info["guessed"])
        action = letter_to_action(letter)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"Guessed  : {letter}")
        print(f"Pattern  : {''.join(info['pattern'])}")
        print(f"Guessed  : {info['guessed']}")
        print(f"Wrong left: {info['wrong_left']}  |  Reward: {reward}")
        print("-" * 40)

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