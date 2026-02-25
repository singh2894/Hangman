from dict import words
from hangman_game import HangmanGame
from heuristic_agent import HeuristicHangmanAgent
import argparse


def run_classic():
    game = HangmanGame(words, max_wrong=6)
    agent = HeuristicHangmanAgent(words)

    state = game.reset()
    done = False

    print("Secret word:", game.secret_word)  # remove for real evaluation
    print()

    while not done:
        pattern, guessed, remaining = state

        print("Pattern:", pattern)
        print("Remaining wrong guesses:", remaining)
        print("Guessed letters:", guessed)

        letter = agent.choose_letter(pattern, guessed)
        print("Agent guesses:", letter)

        state, reward, done = game.guess(letter)
        print("Reward:", reward)
        print("-" * 40)

    if "_" not in game.pattern:
        print("Agent WON 🎉")
    else:
        print("Agent LOST ❌")

    print("Final word:", game.secret_word)


def run_env(seed=42):
    from hangman_env import EnvHangman, WordSampler

    sampler = WordSampler(seed=seed)
    secret_word = sampler.sample(split="train", seed=seed)
    env = EnvHangman(secret_word)
    agent = HeuristicHangmanAgent(words)

    _, info = env.reset(seed=seed)
    done = False

    print("Secret word:", info["word"])  # remove for real evaluation
    print()

    while not done:
        pattern = "".join(ch.lower() for ch in info["pattern"])
        guessed = set(info["guessed"])
        remaining = info["wrong_left"]

        print("Pattern:", pattern)
        print("Remaining wrong guesses:", remaining)
        print("Guessed letters:", guessed)

        letter = agent.choose_letter(pattern, guessed)
        print("Agent guesses:", letter)

        prev_remaining = remaining
        _, terminated, truncated, info = env.step(letter)
        done = terminated or truncated
        reward = 1 if info["wrong_left"] == prev_remaining else -1
        print("Reward:", reward)
        print("-" * 40)

    if "_" not in info["pattern"]:
        print("Agent WON 🎉")
    else:
        print("Agent LOST ❌")

    print("Final word:", info["word"])


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("classic", "env"),
        default="classic",
        help="Choose classic game loop or environment loop.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by env mode.",
    )
    args = parser.parse_args()

    if args.mode == "env":
        run_env(seed=args.seed)
        return
    run_classic()


if __name__ == "__main__":
    run()
