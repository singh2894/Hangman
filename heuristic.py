from collections import Counter
import string


class HeuristicHangmanAgent:

    def __init__(self, dictionary_words):
        self.dictionary = [
            w.lower() for w in dictionary_words
            if len(w) == 5 and w.isalpha()
        ]

        # Step 1: count total letter appearances in dictionary
        self.global_counts = Counter()
        for w in self.dictionary:
            self.global_counts.update(w)

        # Step 2: rank letters by frequency
        self.global_ranked_letters = [
            ch for ch, _ in self.global_counts.most_common()
        ]

    def _filter_candidates(self, pattern, guessed_letters):
        revealed = set(pattern) - {"_"}
        wrong_letters = guessed_letters - revealed

        candidates = []

        for w in self.dictionary:
            match = True

            # Must match revealed pattern
            for i, p in enumerate(pattern):
                if p != "_" and w[i] != p:
                    match = False
                    break

            if not match:
                continue

            # Must not contain wrong letters
            if any(ch in w for ch in wrong_letters):
                continue

            candidates.append(w)

        return candidates

    def choose_letter(self, pattern, guessed_letters):

        candidates = self._filter_candidates(pattern, guessed_letters)

        # Step 3: pick most frequent letter among remaining candidates
        counter = Counter()

        for w in candidates:
            for ch in set(w):
                if ch not in guessed_letters:
                    counter[ch] += 1

        if counter:
            return counter.most_common(1)[0][0]

        # fallback to global ranking
        for ch in self.global_ranked_letters:
            if ch not in guessed_letters and ch in string.ascii_lowercase:
                return ch

        raise RuntimeError("No letters left.")