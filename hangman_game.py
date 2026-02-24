import random


class HangmanGame:
    def __init__(self, word_list, max_wrong=6):
        self.word_list = [w.lower() for w in word_list if len(w) == 5 and w.isalpha()]
        self.max_wrong = max_wrong
        self.reset()

    def reset(self):
        self.secret_word = random.choice(self.word_list)
        self.pattern = ["_"] * 5
        self.guessed_letters = set()
        self.remaining_wrong = self.max_wrong
        return self.get_state()

    def guess(self, letter):
        if letter in self.guessed_letters:
            return self.get_state(), 0, False

        self.guessed_letters.add(letter)

        if letter in self.secret_word:
            for i, ch in enumerate(self.secret_word):
                if ch == letter:
                    self.pattern[i] = letter
            reward = 1
        else:
            self.remaining_wrong -= 1
            reward = -1

        done = self.is_done()
        return self.get_state(), reward, done

    def is_done(self):
        return "_" not in self.pattern or self.remaining_wrong <= 0

    def get_state(self):
        return "".join(self.pattern), self.guessed_letters, self.remaining_wrong