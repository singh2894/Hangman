from dict import words
from hangman_game import HangmanGame
from heuristic_agent import HeuristicHangmanAgent


def run():
    game = HangmanGame(words, max_wrong=6)
    agent = HeuristicHangmanAgent(words)

    state = game.reset()
    done = False

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


if __name__ == "__main__":
    run()