# main.py
from dict import words
from heuristic_agent import HeuristicHangmanAgent
from hangman_env import WordSampler, EnvHangmanGym
import argparse


def letter_to_action(letter: str) -> int:
    letter = letter.lower()
    return ord(letter) - ord("a")


def run_gym(seed=42, split="train"):
    sampler = WordSampler(seed=seed)
    secret_word = sampler.sample(split=split, seed=seed)

    env = EnvHangmanGym(secret_word)
    agent = HeuristicHangmanAgent(words)

    # Debug (keep for now)
    print("Using env class:", env.__class__.__name__)

    obs, info = env.reset(seed=seed)

    print("Secret word:", info.get("word", secret_word))  # remove for real evaluation
    print()

    done = False
    while not done:
        # choose_letter expects tuple + list from info
        letter = agent.choose_letter(info["pattern"], info["guessed"])
        action = letter_to_action(letter)

        out = env.step(action)

        # ✅ Works whether step returns 4 or 5 values
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif len(out) == 4:
            obs, terminated, truncated, info = out
            # fallback reward (since gym reward missing)
            reward = 0.0
        else:
            raise RuntimeError(f"Unexpected number of values from env.step(): {len(out)}")

        done = terminated or truncated

        print("Pattern:", "".join(info["pattern"]))
        print("Wrong left:", info["wrong_left"])
        print("Guessed:", info["guessed"])
        print("Agent guessed:", letter, "| reward:", reward)
        print("-" * 40)

    if "_" not in info["pattern"]:
        print("Agent WON 🎉")
    else:
        print("Agent LOST ❌")

    print("Final word:", info.get("word", secret_word))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()

    run_gym(seed=args.seed, split=args.split)


if __name__ == "__main__":
    run()