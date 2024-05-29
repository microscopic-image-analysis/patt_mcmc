"""
Provides a wrapper for parallelized plain samplers that adhere to a certain
interface (in terms of their arguments and return values) via the function
parallel_plain.
"""

import numpy as np
import numpy.random as rnd
import multiprocessing as mp

def parallel_plain(
        sampler,
        log_density,
        n_chains,
        n_its,
        x_0s,
        hyper=None,
        verbose=True,
        bar=True
    ):
    """A wrapper for parallelized plain samplers that adhere to a certain
        interface (in terms of their arguments and return values).

        Args:
            sampler: underlying plain sampler, must take this function's
                other arguments log_density, n_its, x_0[i], hyper (if not None),
                an instance of rnd.Generator and this function's argument bar
                (in this order!) as its arguments and return the generated
                samples, the generator in its final state, the array of target
                density evaluation counts and the array of iteration-wise
                runtimes (in this order); other returns after these four are
                permitted but will not be processed
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point; must be pickle-
                able (i.e. it mustn't be a lambda function)
            n_chains: number of parallel chains to use, must be positive integer
            n_its: number of iterations to perform per chain, must be positive 
                integer
            x_0s: initial values for the parallel chains, must be 2d np array of
                shape (n_chains,d)
            hyper: hyperparameter(s) of the plain sampler, leave as None if 
                there aren't any
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure (for practical reasons, the 
                progress bar will display the progression of an arbitrary chain,
                not that of the slowest one)

        Returns:
            samples: np array of shape (n_chains, n_its+1, d), where 
                samples[i,j] is the j-th sample generated by the i-th chain
            tde_cnts: 2d np array of size (n_chains,n_its+1), where
                tde_cnts[i,j] is the number of target density evaluations used
                by the i-th chain in the j-th iteration
            runtimes: 2d np array of size (n_chains,n_its+1), where
                runtimes[i,j] is the time the i-th chain took to perform its
                j-th iteration (in seconds)
    """
    if verbose:
        print("Checking validity of given arguments...")
    if type(n_chains) != int or n_chains <= 0:
        raise TypeError("Number of parallel chains must be positive integer!")
    if type(n_its) != int or n_its <= 0:
        raise TypeError("Number of iterations must be positive integer!")
    if type(x_0s) != np.ndarray or len(x_0s.shape) != 2 \
        or x_0s.shape[0] != n_chains:
        raise TypeError("Initial values must be given as 2d np array of shape" \
            + " (n_chains,d)!")
    if verbose:
        print("Preparing for parallel sampling...")
    d = x_0s.shape[-1]
    # initialize generators for random numbers
    seeds = rnd.SeedSequence().spawn(n_chains)
    gens = [rnd.Generator(rnd.MT19937(s)) for s in seeds]
    # prepare arguments for parallelized call
    bars = [bar] + (n_chains - 1) * [False]
    if type(hyper) == type(None):
        starmap_args = [(log_density, n_its, x_0, gen, b)
                        for x_0, gen, b in zip(x_0s, gens, bars)]
    else:
        starmap_args = [(log_density, n_its, x_0, hyper, gen, b)
                        for x_0, gen, b in zip(x_0s, gens, bars)]
    # run parallel chains and process the returns
    if verbose:
        print("Starting parallel sampling...")
    pool = mp.Pool(n_chains)
    returns = pool.starmap(sampler, starmap_args)
    pool.close()
    if verbose:
        print("Processing returns and terminating...")
    samples = np.transpose(np.array([ret[0] for ret in returns]), axes=(1,0,2))
    tde_cnts = np.array([ret[2] for ret in returns]).T
    runtimes = np.array([ret[3] for ret in returns]).T
    return samples, tde_cnts, runtimes

