from typing import Protocol , List , Tuple
from bitstring import Bitstring
from games import Game
import random


# --- Selection Operator Interface ---
class SelectionOperator(Protocol):
    
    name : str
    
    # When called, a selection operator returns a tuple ( x , y , m ), where x
    # is the selected predator, y is the selected prey, and m is the number of
    # function evaluations used to carry out the selection procedure.
    def __call__(
        self, preds: List[Bitstring], preys: List[Bitstring], game: Game
    ) -> Tuple[Bitstring, Bitstring, int]: ...

    # Function alpha for which the conditions of Lemma B.7 are satisfied when
    # the algorithm is applied to zero-sum games with a unique strict pure Nash
    # equilibrium. As the selection operators in this project will treat
    # predators and preys symmetrically, we will assume the sister function
    # beta(sigma,a,b) is equal to alpha(1-sigma,b,a). This is only required in
    # this project for adding plots of the theoretical bounds C3 (Theorem 3.7)
    # and D2.1 (Theorem 3.8) on top of the heatmaps produced by the experiment,
    # and have no bearing on the experiment itself.
    def alpha(self, sigma: float, a: float, b:float) -> float: ...

    # Smallest value of k for which the operator is a k-candidate selection
    # operator (see Theorem 3.6). Also only used for the plotting of theoretical
    # results.
    num_candidates: int



class PairwiseDominanceSelection:
    # Selection operator for PDCoEA, as defined by Algorithm 2.

    def __init__(self):
        self.name = 'PDCoEA'
        self.num_candidates = 2

    def __call__(self, preds, preys, game : Game ):
        x1 , x2 = random.choices(preds,k=2)
        y1 , y2 = random.choices(preys,k=2)
        if game.pred_payoff(x1,y1) >= game.pred_payoff(x2,y1) and game.prey_payoff(x1,y1) >= game.prey_payoff(x1,y2):
            return x1 , y1 , 2
        else:
            return x2 , y2 , 2
        
    def alpha(self,sigma,a,b):
        # As appears near the end of the proof of Theorem 3.7 (Appendix B.6).
        return a*(a+(1-a)*(3*b-b*b+(1-b)*(1-b)*sigma*(1+0.5*sigma)))
    


class TournamentSelection:
    # Selection operator for TSCoEA, as defined by Algorithm 2.

    def __init__(self,k,l):
        self.name = 'TSCoEA'
        self.k = k
        self.l = l
        self.num_candidates = k

    def __call__(self, preds, preys, game : Game ):
        X_pred = random.choices(preds,k=self.k)
        Y_pred = random.choices(preys,k=self.l)
        x = max(X_pred, key=lambda a: min(game.pred_payoff(a,b) for b in Y_pred))
        Y_prey = random.choices(preys,k=self.k)
        X_prey = random.choices(preds,k=self.l)
        y = max(Y_prey, key=lambda b: min(game.prey_payoff(a,b) for a in X_prey))
        return x , y , 2*self.k*self.l
    
    def alpha(self,sigma,a,b):
        # As appears in the proof of Theorem 3.8 (Appendix B.7).
        return a*(2-a-2*(1-a)*(1-b)*(1-b)*sigma)