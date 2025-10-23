from collections.abc import Callable
from games import Game
from selection import SelectionOperator
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.axes as axes
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import differential_evolution
import warnings

# Methods for plotting results of the experiment.


# Custom color map.
cmap = LinearSegmentedColormap.from_list("mycmap", [
    ( 0.00 , '#F7A4A4') ,
    ( 0.35 , '#FEBE8C') ,
    ( 0.99 , '#FFFBC1') ,
    ( 1.00 , '#B6E2A1')
])

# Graph style.
title_fontsize = 26
axis_fontsize = 16
figure_scale = 7

def plot_formula( ax : axes.Axes , formula , color ):
    # Procedure for plotting the curve of a formula.
    x0 , x1 = ax.get_xlim()
    x_vals = np.linspace(x0,x1,1000)
    y_vals = [ formula(x) for x in x_vals ]
    ax.plot(x_vals,y_vals,color=color,linewidth=2,alpha=0.5)


def stable_bound(alpha):
    # Procedure for computing the curve q_0 as defined in C3 (Theorem 3.7) and
    # D2 (Theorem 3.8). (See also Lemma B.7.)
    def f(gamma, a, sigma):
        denom = alpha(sigma, a, gamma / a) * alpha(1 - sigma, gamma / a, a)
        # Avoid division by zero or NaN
        if np.isnan(denom) or denom == 0:
            return np.nan
        return gamma / denom
    def bound(gamma):
        def neg_f2d(z):
            a, sigma = z
            if not (gamma <= a <= 1 and 0 <= sigma <= 1):
                return np.inf
            try:
                val = f(gamma, a, sigma)
                # If val is NaN, make it bad for optimization
                if np.isnan(val) or np.isinf(val):
                    return np.inf
                return -val
            except Exception:
                return np.inf

        bounds = [(gamma, 1), (0, 1)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in subtract")
            result = differential_evolution(neg_f2d, bounds, polish=False)
        sup = -result.fun if result.success else np.nan
        return sup
    return bound

def unstable_bound(k):
    # For computing the threshold in B1 of Theorem 3.6.
    def bound(gamma):
        if gamma == 0:
            return 1/k
        else:
            return gamma / (1-(1-gamma)**k)
    return bound



def draw_grid( ax : axes.Axes , selection_operator: SelectionOperator, num_rows : int , num_cols : int , colour_fn : Callable[[int,int],float] ):
    # Function that plots the experiment results on a given axes for a single
    # Game-SelectionOperator combination, with the proportion of stable runs
    # being specified by colour_fn.
    row_height = 1/num_rows
    col_width = 1/num_cols
    for i in range(num_cols):
        for j in range(num_rows):
            rect = patches.Rectangle(
                (i*col_width,j*row_height),
                col_width, row_height,
                facecolor = cmap(colour_fn(i,j)))
            ax.add_patch(rect)
    # Set limits and label axes.
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel(r'$\gamma$',fontsize=axis_fontsize)
    ax.set_ylabel(r'$q$',fontsize=axis_fontsize,rotation=0)
    ax.set_aspect('equal')
    # Add known theoretical bounds for the given selection operator.
    plot_formula(ax,unstable_bound(selection_operator.num_candidates),'red')
    plot_formula(ax,stable_bound(selection_operator.alpha),'green')

def create_plot(games: List[Game], selectors: List[SelectionOperator], num_rows, num_cols, colour_fn: Callable[[str, str, int, int], float]):
    # Procedure that saves a figure akin to Figure 2.

    # Get number of games and selection operators in experiment and set figure
    # size.
    plot_nrows = len(selectors)
    plot_ncols = len(games)
    fig_width = plot_ncols * figure_scale
    fig_height = plot_nrows * figure_scale
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        plot_nrows,
        plot_ncols + 1,
        width_ratios=[1]*plot_ncols + [0.05],
        wspace=0.3
    )
    axs = np.empty((plot_nrows, plot_ncols), dtype=object)

    # Add results to each axes.
    for i in range(plot_nrows):
        for j in range(plot_ncols):
            s = selectors[i]
            g = games[j]
            def local_colour_fn(k, l):
                return colour_fn(g.name, s.name, k, l)
            axs[i, j] = fig.add_subplot(gs[i, j])
            draw_grid(axs[i, j], s, num_rows, num_cols, local_colour_fn)

    # Formatting for axes.
    hpad = 25
    vpad = 35
    for ax, alg in zip(axs[:, 0], selectors):
        ax.annotate(alg.name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - hpad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=title_fontsize, ha='right', va='center')
    for ax, game in zip(axs[0, :], games):
        ax.annotate(game.name, xy=(0.5, 1), xytext=(0, vpad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=title_fontsize, ha='center', va='baseline')
    
    # Add the color bar to the side.
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap),
        cax=cax,
        orientation='vertical'
    )
    cbar.ax.set_ylabel('Proportion of Stable Runs', fontsize=axis_fontsize)
    
    # Save figure.
    output_filename = "plot.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)