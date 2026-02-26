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
LETTERS   = [chr(ord('a') + i) for i in range(26)]


# State encoding / decoding 

def _encode_state(pattern: tuple, guessed: frozenset, wrong_left: int) -> str:
    """Encode (p, G, w) → "pattern=PL___|guessed=ALP|wrong_left=4" """
    return (f"pattern={''.join(pattern)}"
            f"|guessed={''.join(sorted(guessed)).upper()}"
            f"|wrong_left={wrong_left}")

def _decode_state(state_str: str) -> tuple:
    """Decode state string → (pattern tuple, guessed frozenset, wrong_left int)"""
    p = dict(tok.split("=") for tok in state_str.split("|"))
    return tuple(p["pattern"]), frozenset(p["guessed"].lower()), int(p["wrong_left"])


# On-the-fly transition helper 

def _apply_action(word: str, pattern: tuple, guessed: frozenset,
                  wrong_left: int, action: str) -> tuple:
    """
    Compute next (pattern, guessed, wrong_left) directly from current state + action.
    Replaces BFS — called inside step() each time an action is taken.
    """
    new_guessed = guessed | {action}
    if action in word:
        # Reveal all positions where this letter appears
        new_pattern = tuple(action.upper() if word[i] == action else pattern[i]
                            for i in range(len(word)))
        new_wrong_left = wrong_left          # correct guess: no life lost
    else:
        new_pattern    = pattern             # wrong guess: pattern unchanged
        new_wrong_left = wrong_left - 1      # lose one wrong guess
    return new_pattern, new_guessed, new_wrong_left


# Word sampler: 80/20 train-test split 

class WordSampler:
    """Deterministic 80/20 train/test split of WORD_LIST.

    Usage: sampler = WordSampler(seed=42)
           word    = sampler.sample(split="train", seed=0)
    """
    def __init__(self, seed: int = 42, train_ratio: float = 0.8):
        rng = random.Random(seed)
        shuffled = WORD_LIST[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * train_ratio)
        self.train_words = shuffled[:cut]   # ≈ 80%
        self.test_words  = shuffled[cut:]   # ≈ 20%

    def sample(self, split: str = "train", seed: Optional[int] = None) -> str:
        pool = self.train_words if split == "train" else self.test_words
        return random.Random(seed).choice(pool)


class EnvHangman:
    """
    Plain-Python Hangman env. No BFS — transitions computed on the fly.
    step() returns (obs, terminated, truncated, info) — no reward.
    obs = state string e.g. "pattern=PL___|guessed=ALP|wrong_left=4"
    """

    def __init__(self, word: str):
        self._word         = word.lower()
        self._pattern      = None   # tuple  e.g. ("H","E","_","_","_")
        self._guessed      = None   # frozenset  e.g. frozenset({"h","e"})
        self._wrong_left   = None   # int
        self.np_random     = None   # numpy random generator
        self._step_counter = None   # int

    def _get_obs(self) -> str:
        """State string — used directly as Q-table key by Person 3."""
        return _encode_state(self._pattern, self._guessed, self._wrong_left)

    def _get_info(self) -> dict:
        return {
            "word":       self._word,
            "pattern":    self._pattern,
            "masked":     " ".join(self._pattern),   # e.g. "H E _ _ _"
            "guessed":    sorted(self._guessed),
            "wrong_left": self._wrong_left,
            "step":       self._step_counter,
        }

    def reset(self, seed: Optional[int] = None):
        """Returns (obs, info). Initialises to all-hidden, full lives."""
        self.np_random     = np.random.default_rng(seed=seed)
        self._pattern      = tuple("_" * len(self._word))
        self._guessed      = frozenset()
        self._wrong_left   = MAX_WRONG
        self._step_counter = 0
        return self._get_obs(), self._get_info()

    def step(self, action: str):
        """Takes a letter. Returns (obs, terminated, truncated, info). (No reward)."""
        # Illegal action: letter already guessed → state unchanged
        if action in self._guessed:
            return self._get_obs(), False, False, self._get_info()
        
        if action in self._word:
            reward = 1    # correct guess
        else:
            reward = -1   # wrong guess

        # Compute next state on the fly — no lookup needed
        self._pattern, self._guessed, self._wrong_left = _apply_action(
            self._word, self._pattern, self._guessed, self._wrong_left, action
        )
        self._step_counter += 1
        info       = self._get_info()
        terminated = "_" not in self._pattern or self._wrong_left == 0
        truncated  = self._step_counter >= 500
        # Win bonus: all letters revealed
        if "_" not in self._pattern:
            reward += 5

        return self._get_obs(), reward, terminated, truncated, info

    def _get_legal_actions(self) -> list[str]:
        """A(s) = all letters not yet guessed."""
        return [a for a in LETTERS if a not in self._guessed]


GymBase = gym.Env if gym is not None else object


class EnvHangmanGym(GymBase):

    def __init__(self, word: str):
        if gym is None:
            raise ModuleNotFoundError("gymnasium is required to use EnvHangmanGym.")
        super().__init__()
        self._word = word.lower()

        self.observation_space = gym.spaces.Box(
            low=0, high=26, shape=(32,), dtype=np.int32
        )

        # Action: 26 letters
        self.action_space      = gym.spaces.Discrete(26)
        self._action_to_letter = {i: chr(ord('a') + i) for i in range(26)}

        self._pattern      = None
        self._guessed      = None
        self._wrong_left   = None
        self._step_counter = None

    def _encode_obs(self) -> np.ndarray:

        obs = np.zeros(32, dtype=np.int32)
        for i, ch in enumerate(self._pattern):
            obs[i] = 0 if ch == "_" else ord(ch.lower()) - ord('a') + 1
        for i, letter in enumerate(LETTERS):
            obs[5 + i] = 1 if letter in self._guessed else 0
        obs[31] = self._wrong_left
        return obs

    def _get_obs(self) -> np.ndarray:
        return self._encode_obs()

    def _get_info(self) -> dict:
        return {
            "word":       self._word,
            "pattern":    self._pattern,
            "masked":     " ".join(self._pattern),
            "guessed":    sorted(self._guessed),
            "wrong_left": self._wrong_left,
            "step":       self._step_counter,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)   # seeds self.np_random
        self._pattern      = tuple("_" * len(self._word))
        self._guessed      = frozenset()
        self._wrong_left   = MAX_WRONG
        self._step_counter = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            action = self._action_to_letter[int(action)]

        # Illegal action: letter already guessed → state unchanged
        if action in self._guessed:
            return self._get_obs(), False, False, self._get_info()
        
        # Compute reward before transition
        if action in self._word:
            reward = 1    # correct guess
        else:
            reward = -1   # wrong guess

        # Compute next state on the fly — no lookup needed
        self._pattern, self._guessed, self._wrong_left = _apply_action(
            self._word, self._pattern, self._guessed, self._wrong_left, action
        )
        self._step_counter += 1
        info       = self._get_info()
        terminated = "_" not in self._pattern or self._wrong_left == 0
        truncated  = self._step_counter >= 500

         # Win bonus: all letters revealed
        if "_" not in self._pattern:
            reward += 5

        return self._get_obs(), reward, terminated, truncated, info

    def _get_legal_actions(self) -> list[str]:
        """A(s) = all letters not yet guessed."""
        return [a for a in LETTERS if a not in self._guessed]

    @property
    def n_actions(self) -> int:
        """Always 26 — one per letter of the alphabet."""
        return 26

    def check(self):
        if check_env is None:
            print("gymnasium is not installed; skipping environment checks.")
            return
        try:
            check_env(self); print("Environment passes all checks!")
        except Exception as e:
            print(f"Environment has issues: {e}")


class PolicyRandom:
    def __init__(self): pass

    def _get_action(self, env):
        """Uniformly random legal action."""
        actions = env._get_legal_actions()
        return str(env.np_random.choice(actions)) if actions else None
