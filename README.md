# Optimal transport with f-divergence regularization and generalized Sinkhorn algorithm

This is the official codebase for the paper "Optimal transport with f-divergence regularization and generalized Sinkhorn algorithm" by Dávid Terjék and Diego González-Sánchez accepted for publication at the 25<sup>th</sup> International Conference on Artificial Intelligence and Statistics (AISTATS) 2022.

Before running the experiments, prepare a python 3 environment with the following packages:
* argparse
* imageio
* matplotlib
* numpy
* tensorboard
* torch

And download the data from https://github.com/jeanfeydy/global-divergences/tree/master/sinkhorn_entropies/data

To run the experiments, execute

```python fot_experiment.py --log_dir <path to a directory where plots will be saved> --data_dir <path to a directory where the supplied data (.png files) can be found>```

The following parameters can be used for configuration:

* ```--random_seed <integer: fixes the seed of random number generation, influences pointcloud sampling from the data>```
* ```--mu_size <integer: number of points in the red pointcloud>```
* ```--nu_size <integer: number of points in the blue pointcloud>```
* ```--tolerance <float: convergence of Sinkhorn is assumed when this tolerance level is reached>```
* ```--epsilon <float: etropic regularization coefficient>```
* ```--dataset <string: one of "moons", "densities", "slopes", "crescents">```
* ```--divergence <string: one of "kl", "reverse_kl", "chi2", "reverse_chi2", "hellinger2", "js", "jeffreys", "triangular">```
* ```--double or --float: sets single or double precision```