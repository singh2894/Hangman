# heuristic_agent.py
from collections import Counter
import string


class HeuristicHangmanAgent:
    def __init__(self, dictionary_words):
        self.dictionary = [
            w.lower() for w in dictionary_words
            if len(w) == 5 and w.isalpha()
        ]

        self.global_counts = Counter()
        for w in self.dictionary:
            self.global_counts.update(w)

        self.global_ranked_letters = [ch for ch, _ in self.global_counts.most_common()]

    @staticmethod
    def _pattern_tuple_to_str(pattern_tuple) -> str:
        # env gives ('A','_','_','E','_') -> "a__e_"
        return "".join("_" if ch == "_" else str(ch).lower() for ch in pattern_tuple)

    def _filter_candidates(self, pattern_str: str, guessed_letters: set[str]):
        revealed = set(pattern_str) - {"_"}
        wrong_letters = guessed_letters - revealed

        candidates = []
        for w in self.dictionary:
            # match revealed positions
            ok = True
            for i, p in enumerate(pattern_str):
                if p != "_" and w[i] != p:
                    ok = False
                    break
            if not ok:
                continue

            # must not contain wrong letters
            if any(ch in w for ch in wrong_letters):
                continue

            candidates.append(w)

        return candidates

    def choose_letter(self, pattern_tuple, guessed_list):
        pattern_str = self._pattern_tuple_to_str(pattern_tuple)
        guessed_letters = {str(g).lower() for g in guessed_list}

        candidates = self._filter_candidates(pattern_str, guessed_letters)

        counter = Counter()
        for w in candidates:
            for ch in set(w):
                if ch not in guessed_letters:
                    counter[ch] += 1

        if counter:
            return counter.most_common(1)[0][0]

        for ch in self.global_ranked_letters:
            if ch not in guessed_letters and ch in string.ascii_lowercase:
                return ch

        raise RuntimeError("No letters left.")