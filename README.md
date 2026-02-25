# Hangman


## Problem description (Hangman, 5-letter English words)

Hangman is a sequential decision game between an agent (guesser) and the environment (word generator).

At the start of each episode, the environment samples a secret word w uniformly (or from a chosen distribution) from a fixed dictionary 
𝑊 W of 5-letter English words.

The agent repeatedly guesses one letter at a time.

If the guessed letter appears in w, all occurrences of that letter are revealed immediately (in the correct positions). The number of remaining wrong guesses does not change.

If the guessed letter does not appear in w, the number of remaining wrong guesses decreases by 1.

The game ends when:

the agent reveals the entire word (win), or
the remaining wrong guesses reaches 0 (lose).

## MDP formulation

We define the MDP as the tuple (S,A,P,R,γ).
Alphabet and dictionary

Alphabet (lowercase English letters):
Σ={a,b,…,z},∣Σ∣=26.

Dictionary of valid secret words:
W⊆Σ^5.

## How to run

Run the default classic loop:

```bash
python3 main.py
```

Run the environment loop:

```bash
python3 main.py --mode env
```

Use a fixed seed in environment mode:

```bash
python3 main.py --mode env --seed 42
```
