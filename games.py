from typing import Protocol
from bitstring import Bitstring
import numpy as np
import bitstring


# --- Game Interface ---
# (see Definition 2.1)
class Game(Protocol):
    name : str

    # We assume both the predator domain X and prey domain Y are equal to
    # {0,1}^n (that is, bitstrings of length n). n is specified here.
    n: int

    # Payoff function for predator.
    def pred_payoff(self, x: Bitstring, y: Bitstring) -> float: ...

    # Payoff function for prey. For zero-sum games, we will take
    # prey_payoff(x,y) = -pred_payoff(x,y)
    def prey_payoff(self, x: Bitstring, y: Bitstring) -> float: ...

    # All games defined in this project are members of the class of games with a
    # unique strict pure Nash equilibrium. The pure strategies that constitute
    # this Nash equilibrium are defined below.
    optimal_pred: Bitstring
    optimal_prey: Bitstring



class Bilinear(Game):
    # Bilinear game as defined in Appendix C.1.
    def __init__(self, n : int, a : int , b : int):
        self.name = 'Bilinear'
        self.n = n
        self.a = a
        self.b = b
        self.optimal_pred = np.array([True] * a + [False] * (n-a))
        self.optimal_prey = np.array([True] * b + [False] * (n-b))
    
    def h(self,x):
        return bitstring.leading_ones(x) + bitstring.trailing_zeros(x)
    def pred_payoff(self, x: Bitstring, y: Bitstring):
        norm_x = bitstring.norm(x)
        norm_y = bitstring.norm(y)
        if norm_x == self.a and norm_y != self.b:
            return 0.5 + self.h(x) - self.h(y)
        elif norm_x != self.a and norm_y == self.b:
            return -0.5 + self.h(x) - self.h(y)
        else:
            return (norm_x-self.a)*(norm_y-self.b)+self.h(x)-self.h(y)
    def prey_payoff(self, x, y):
        return -self.pred_payoff(x,y)
    


class PlantedBilinear(Game):
    # PlantedBilinear game as defined in Appendix C.2.
    def __init__(self, n : int):
        self.name = 'Planted Bilinear'
        self.n = n
        rng = np.random.default_rng(seed=0)
        self.u = bitstring.seeded_random(n, rng=rng)
        self.v = bitstring.seeded_random(n, rng=rng)
        self.A = rng.random((n, n))
        self.eps = np.nextafter(0,1)
        self.optimal_pred = self.u
        self.optimal_prey = self.v
    
    def A_product(self, x: Bitstring, y: Bitstring):
        x_f = x.astype(float)
        y_f = y.astype(float)
        return x_f @ self.A @ y_f
    
    def pred_payoff(self, x: Bitstring, y: Bitstring):
        if bitstring.equal(x,self.u) and bitstring.equal(y,self.v):
            return 0.
        elif bitstring.equal(x,self.u):
            return self.eps
        elif bitstring.equal(y,self.v):
            return -self.eps
        else:
            return 2 * self.A_product(x,y) - self.A_product(x,self.v) - self.A_product(self.u,y)
    def prey_payoff(self, x, y):
        return -self.pred_payoff(x,y)
    


class MBJR_2024(Game):
    # Game first introduced by Maiti, Boczar, Jamieson, and Ratliff, as defined
    # in Appendix C.3.
    def __init__(self, n : int , delta_min : float , delta_1 : float):
        self.name = 'MBJR_2024'
        self.n = n
        self.delta_min = delta_min
        self.delta_1 = delta_1
        self.optimal_pred = np.ones(n, dtype=bool)
        self.optimal_prey = np.ones(n, dtype=bool)
    
    def pred_payoff(self, x: Bitstring, y: Bitstring):
        norm_x = bitstring.norm(x)
        norm_y = bitstring.norm(y)
        if norm_x == self.n and norm_y == self.n:
            return 0.
        elif norm_x == self.n and norm_y == self.n-1:
            return 2*self.delta_min
        elif norm_x == self.n-1 and norm_y == self.n:
            return -2*self.delta_min
        elif norm_x == self.n:
            return 2*self.delta_1
        elif norm_y == self.n:
            return -2*self.delta_1
        elif norm_x == norm_y:
            return 0.
        elif norm_y < norm_x:
            return 1.
        else:
            return -1.
    def prey_payoff(self, x, y):
        return -self.pred_payoff(x,y)
