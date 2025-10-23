from typing import TypeAlias
import numpy as np

# Basic methods for working with bitstrings.

Bitstring: TypeAlias = np.ndarray[np.bool_]

def norm(x: Bitstring):
    # Number of 1-bits in x.
    return int(np.sum(x))

def leading_ones(x: Bitstring):
    # Longest prefix of 1s in x.
    if np.all(x):
        return len(x)
    return int(np.argmax(x == 0))

def trailing_zeros(x: Bitstring):
    # Longest suffix of 0s in x.
    if not np.any(x):
        return len(x)
    return int(np.argmax(x[::-1] == 1))

def equal(x: Bitstring, y: Bitstring):
    # Tests whether bitstrings x and y are equal.
    return (x==y).all()

def random(n):
    # Generates a bitstring of length n uniformly at random.
    return np.random.choice(a=[False, True], size=(n,))

def seeded_random(n, rng):
    # Generates a bitstring of length n uniformly at random using a given random
    # seed.
    return rng.choice(a=[False, True], size=(n,))

def mutate(x: Bitstring, chi: float):
    # Flips each bit of x independently with probability chi/n, where n is the
    # length of x.
    p = chi/len(x)
    flip_mask = np.random.random(size=x.shape) < p
    return np.logical_xor(x, flip_mask)