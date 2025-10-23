import ray
from games import Game , Bilinear , PlantedBilinear , MBJR_2024
from selection import SelectionOperator , PairwiseDominanceSelection , TournamentSelection
from bitstring import Bitstring
from typing import List
import numpy as np
import random
from collections import defaultdict
import bitstring
import plotting
import progress


###############################################################################
################### Specify parameters for experiment here. ###################
###############################################################################
# Experiment parameters, game definitions, and selection operators listed here
# are the same as those used to generate Figure 2.
pop_size = 100
num_rows = 10 # Number of different values of q to test.
num_cols = 10 # Number of different values of gamma to test.
runs_per_cell = 25
max_fevals_per_run = 4000
games = [
    Bilinear(n=50,a=35,b=10),
    PlantedBilinear(n=50),
    MBJR_2024(n=50,delta_min=0.001,delta_1=0.1)]
selectors = [
    PairwiseDominanceSelection(),
    TournamentSelection(2,2)]
###############################################################################
###############################################################################
###############################################################################



def probabilistic_round(x):
    return int(np.floor(x + random.random()))

def generate_initial_populations(gamma: float, game: Game):
    # For a given game, generates an initial populations with a gamma proportion
    # of bitstrings on that are optimal.
    a = random.uniform(0,1)
    num_optimal_preds = probabilistic_round((gamma ** a) * pop_size)
    num_optimal_preys = probabilistic_round((gamma ** (1-a)) * pop_size)
    def initial_pred(i):
        if i < num_optimal_preds:
            return game.optimal_pred
        else:
            return bitstring.random(game.n)
    def initial_prey(i):
        if i < num_optimal_preys:
            return game.optimal_prey
        else:
            return bitstring.random(game.n)
    preds = [initial_pred(i) for i in range(pop_size)]
    preys = [initial_prey(i) for i in range(pop_size)]
    random.shuffle(preds)
    random.shuffle(preys)
    return preds , preys

def proportion_optimal(preds: List[Bitstring], preys: List[Bitstring], game: Game):
    # Measures the number pairs in pred x prey that lie on the strict Nash
    # equilibrium for the given game.
    num_optimal_preds = sum(1 for pred in preds if bitstring.equal(pred,game.optimal_pred))
    num_optimal_preys = sum(1 for prey in preys if bitstring.equal(prey,game.optimal_prey))
    return (num_optimal_preds/len(preds)) * (num_optimal_preys/len(preys))

def step_algorithm(preds: List[Bitstring], preys: List[Bitstring], game: Game, sel: SelectionOperator, chi: float):
    # Runs a single generation of Algorithm 1.

    # Select pop_size many predator-prey pairs.
    selections = [sel(preds,preys,game) for _ in range(pop_size)]

    # Record the number of function evaluations used across all selections.
    fevals = sum([k for _,_,k in selections])

    # Mutate each selected predator to obtain the next predator population.
    preds = [bitstring.mutate(pred,chi) for pred,_,_ in selections]

    # Mutate each selected prey to obtain the next prey population.
    preys = [bitstring.mutate(prey,chi) for _,prey,_ in selections]
    return preds , preys , fevals



@ray.remote
def single_run(row: int, col: int, gamma: float, gamma_tolerance: float, q: float, game: Game, sel: SelectionOperator):

    # Determine the value of the mutation rate chi that such that mutating
    # bitstrings x and y (both of length n) incurs no change with probability q.
    chi = game.n * (1 - (q ** (1/(2*game.n))))

    # Generate initial populations such that the proportion of optimal
    # predator-prey pairs is within gamma_tolerance of gamma.
    initial_gamma = 2
    while abs(initial_gamma-gamma) > gamma_tolerance:
        preds , preys = generate_initial_populations(gamma, game)
        initial_gamma = proportion_optimal(preds,preys,game)
    
    # Run the algorithm. The variable success will record whether the algorithm
    # demonstated stable behaviour (True) or unstable behaviour (False) for the
    # given run.
    feval_count = 0
    success = False
    while feval_count < max_fevals_per_run:
        preds , preys , fevals = step_algorithm(preds,preys,game,sel,chi)
        feval_count += fevals
        if proportion_optimal(preds,preys,game) < initial_gamma:
            # If the proportion of optimal pairs has dropped below the initial
            # proportion, then we can terminate the run with success = False.
            success = False
            break
    else:
        # Once the algorithm has used max_fevals_per run function evaluations
        # without the proportion of optimal predator-prey pairs dropping below
        # the initial proportion, we can terminate the run with success = True.
        success = True
        
    # Return a dictionary with all relevant data.
    return {
        "row": row,
        "col": col,
        "game": game.name,
        "selector": sel.name,
        "gamma": gamma,
        "q": q,
        "success": success
    }



def sweep():

    # Set up ranges for q and gamma.
    gamma_range = np.linspace( 1/(2*num_cols) , 1 + 1/(2*num_cols) , num=num_cols , endpoint=False)
    gamma_tolerance = 1/(2*num_cols)
    q_range = np.linspace( 1/(2*num_rows) , 1 + 1/(2*num_rows) , num=num_rows , endpoint=False)

    # Use ray to collect results from all required runs.
    futures = [
        single_run.remote(col, row, gamma_range[col], gamma_tolerance, q_range[row], g, s)
            for _ in range(runs_per_cell)
            for col in range(num_cols)
            for row in range(num_rows)
            for g in games
            for s in selectors
    ]

    # Progress bar setup
    res_list = []
    update, close = progress.smart_progress(len(futures), desc="Running simulations")

    while futures:
        done, futures = ray.wait(futures, num_returns=1)
        res_list.extend(ray.get(done))
        update(len(done))

    close()

    # Collect results into a dictionary.
    results = defaultdict(list)  # keys = (game_name, selector_name, gamma, q)
    for res in res_list:
        key = (res["game"], res["selector"], res["row"], res["col"])
        results[key].append(res["success"])

    # Define a function that reads from the dictionary to return the proportion
    # of stable runs for a given game g and selection operator s in the grid
    # cell corresponding to coordinates (i,j).
    def colour_fn(g,s,i,j):
        runs = results[(g,s,i,j)]
        return sum(runs) / len(runs)
    
    # Use plotting module to produce the figure.
    plotting.create_plot(games , selectors , num_rows , num_cols , colour_fn )



if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    sweep()
    ray.shutdown()