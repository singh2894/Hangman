# hangman_env.py
from typing import Optional
import numpy as np
import random

try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
except ModuleNotFoundError:
    gym = None
    check_env = None

from dict import words as WORD_LIST

MAX_WRONG = 6
LETTERS = [chr(ord("a") + i) for i in range(26)]


def _encode_state(pattern: tuple, guessed: frozenset, wrong_left: int) -> str:
    return (
        f"pattern={''.join(pattern)}"
        f"|guessed={''.join(sorted(guessed)).upper()}"
        f"|wrong_left={wrong_left}"
    )


def _decode_state(state_str: str) -> tuple:
    p = dict(tok.split("=") for tok in state_str.split("|"))
    return tuple(p["pattern"]), frozenset(p["guessed"].lower()), int(p["wrong_left"])


def _apply_action(word: str, pattern: tuple, guessed: frozenset,
                  wrong_left: int, action: str) -> tuple:
    new_guessed = guessed | {action}
    if action in word:
        new_pattern = tuple(
            action.upper() if word[i] == action else pattern[i]
            for i in range(len(word))
        )
        new_wrong_left = wrong_left
    else:
        new_pattern = pattern
        new_wrong_left = wrong_left - 1
    return new_pattern, new_guessed, new_wrong_left


class WordSampler:
    """Deterministic 80/20 train/test split of WORD_LIST.

    Usage:
        sampler = WordSampler(seed=42)
        word = sampler.sample(split="train", seed=42)
    """

    def __init__(self, seed: int = 42, train_ratio: float = 0.8):
        rng = random.Random(seed)
        shuffled = WORD_LIST[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * train_ratio)
        self.train_words = shuffled[:cut]
        self.test_words = shuffled[cut:]

    def sample(self, split: str = "train", seed: Optional[int] = None) -> str:
        pool = self.train_words if split == "train" else self.test_words
        # deterministic per call if seed is provided
        return random.Random(seed).choice(pool)


class EnvHangman:
    """
    Plain-Python Hangman env (non-gym).
    reset() -> (obs, info)
    step()  -> (obs, terminated, truncated, info)  # no reward here
    """

    def __init__(self, word: str):
        self._word = word.lower()
        self._pattern = None
        self._guessed = None
        self._wrong_left = None
        self.np_random = None
        self._step_counter = None

    def _get_obs(self) -> str:
        return _encode_state(self._pattern, self._guessed, self._wrong_left)

    def _get_info(self) -> dict:
        return {
            "word": self._word,
            "pattern": self._pattern,              # tuple
            "masked": " ".join(self._pattern),
            "guessed": sorted(self._guessed),      # list
            "wrong_left": self._wrong_left,
            "step": self._step_counter,
        }

    def reset(self, seed: Optional[int] = None):
        self.np_random = np.random.default_rng(seed=seed)
        self._pattern = tuple("_" * len(self._word))
        self._guessed = frozenset()
        self._wrong_left = MAX_WRONG
        self._step_counter = 0
        return self._get_obs(), self._get_info()

    def step(self, action: str):
        action = str(action).lower()

        if action in self._guessed:
            terminated = False
            truncated = False
            return self._get_obs(), terminated, truncated, self._get_info()

        self._pattern, self._guessed, self._wrong_left = _apply_action(
            self._word, self._pattern, self._guessed, self._wrong_left, action
        )
        self._step_counter += 1

        terminated = ("_" not in self._pattern) or (self._wrong_left == 0)
        truncated = self._step_counter >= 500
        return self._get_obs(), terminated, truncated, self._get_info()

    def _get_legal_actions(self) -> list[str]:
        return [a for a in LETTERS if a not in self._guessed]


GymBase = gym.Env if gym is not None else object


class EnvHangmanGym(GymBase):
    """
    Gymnasium-compatible env.
    reset() -> (obs, info)
    step()  -> (obs, reward, terminated, truncated, info)   ✅ ALWAYS 5 values
    """

    def __init__(self, word: str):
        if gym is None:
            raise ModuleNotFoundError("gymnasium is required to use EnvHangmanGym.")
        super().__init__()
        self._word = word.lower()

        # Fixed 5-letter word assumption -> 32-length numeric observation
        self.observation_space = gym.spaces.Box(low=0, high=26, shape=(32,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(26)
        self._action_to_letter = {i: chr(ord("a") + i) for i in range(26)}

        self._pattern = None
        self._guessed = None
        self._wrong_left = None
        self._step_counter = None

    def _encode_obs(self) -> np.ndarray:
        obs = np.zeros(32, dtype=np.int32)

        # pattern: positions 0..4 (0 unknown, 1..26 letters)
        for i, ch in enumerate(self._pattern):
            obs[i] = 0 if ch == "_" else (ord(ch.lower()) - ord("a") + 1)

        # guessed flags: positions 5..30
        for i, letter in enumerate(LETTERS):
            obs[5 + i] = 1 if letter in self._guessed else 0

        # wrong_left: position 31
        obs[31] = self._wrong_left
        return obs

    def _get_obs(self) -> np.ndarray:
        return self._encode_obs()

    def _get_info(self) -> dict:
        return {
            "word": self._word,
            "pattern": self._pattern,              # tuple
            "masked": " ".join(self._pattern),
            "guessed": sorted(self._guessed),      # list
            "wrong_left": self._wrong_left,
            "step": self._step_counter,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._pattern = tuple("_" * len(self._word))
        self._guessed = frozenset()
        self._wrong_left = MAX_WRONG
        self._step_counter = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        ✅ Gymnasium step must return 5 values:
            obs, reward, terminated, truncated, info
        """
        # Convert int action -> letter
        if isinstance(action, (int, np.integer)):
            letter = self._action_to_letter[int(action)]
        else:
            letter = str(action).lower()

        # Repeated guess: no state change, neutral reward
        if letter in self._guessed:
            reward = 0.0
            terminated = False
            truncated = False
            obs = self._get_obs()
            info = self._get_info()
            return obs, reward, terminated, truncated, info

        prev_wrong_left = self._wrong_left

        self._pattern, self._guessed, self._wrong_left = _apply_action(
            self._word, self._pattern, self._guessed, self._wrong_left, letter
        )
        self._step_counter += 1

        # reward: +1 if correct (no life lost), else -1
        reward = 1.0 if self._wrong_left == prev_wrong_left else -1.0

        terminated = ("_" not in self._pattern) or (self._wrong_left == 0)
        truncated = self._step_counter >= 500

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_legal_actions(self) -> list[str]:
        return [a for a in LETTERS if a not in self._guessed]

    @property
    def n_actions(self) -> int:
        return 26

    def check(self):
        if check_env is None:
            print("gymnasium is not installed; skipping environment checks.")
            return
        try:
            check_env(self)
            print("Environment passes all checks!")
        except Exception as e:
            print(f"Environment has issues: {e}")