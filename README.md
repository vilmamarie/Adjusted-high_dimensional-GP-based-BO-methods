# Adjusted high-dimensional GP-based BO methods
This repository contains implementations for high-dimensional GP-based Bayesian Optimization methods Random Embedding Bayesian Optimization (REMBO), Hashing-enhanced Subspace Bayesian Optimization (HeSBO), and Bayesian Optimization with Cylindrical Kernels (BOCK), adjusted to work with additional benchmark and acquisition functions.

The code for REMBO and HeSBO is taken from https://github.com/aminnayebi/HesBO, and adjusted to include the Ackley, Griewank, Levy, Levy N.13, Schwefel, and 3-dimensional Hartmann Benchmark functions (https://www.sfu.ca/~ssurjano/optimization.html), and the Probability of Improvement acquisition function (https://distill.pub/2020/bayesian-optimization/).

The code for BOCK is taken from https://github.com/QUVA-Lab/HyperSphere, and adjusted to include the Ackley, Griewank, Levy N.13, and 3-dimensional Hartmann Benchmark functions (https://www.sfu.ca/~ssurjano/optimization.html), and the Probability of Improvement acquisition function (https://distill.pub/2020/bayesian-optimization/).
