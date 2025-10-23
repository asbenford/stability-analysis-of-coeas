# Stability Analysis of Coevolutionary Algorithms

This project contains a Python implementation of the experiments used to carry out the empirical analysis in a forthcoming NeurIPS paper \[1\]. Running ``experiment.py`` produces a figure containing heatmaps showing how algorithmic stability depends on mutation strength for a range of selection operators and games.

``experiment.py`` begins with a number of variable assignments that deterimine the experiment specification, including parameters for
* population size,
* the number of cells in each row/column of the heatmap,
* the number of runs per cell,
* the maximum number function evaluations per run,
* a list of games to analyse
* a list of selection operators to analyse.


## Ray framework

The default experiment specification in ``experiment.py`` is that used to produce Figure 2 of \[1\]. However, this specification is highly intensive and required use of an HPC cluster to execute practically. While the specification can be edited as desired, this project has been implemented using [Ray](https://docs.ray.io/en/latest/index.html) so that the full experiment can be faithfully reproduced on a range of architectures. For guidance on deployment to clusters, see Ray documentation [here](https://docs.ray.io/en/latest/cluster/getting-started.html#cluster-index).

## References

\[1\] A. Benford and P. K. Lehre. Theoretical Guarantees for the Retention of Strict Nash Equilibria by Coevolutionary Algorithms. In *Advances in Neural Information Processing Systems 38*, NeurIPS '25, 2025.
