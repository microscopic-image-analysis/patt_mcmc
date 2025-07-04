"""
Provides implementations of various different versions of elliptical slice
sampling (ESS). For the user's convenience, we make a distinction between ESS
applied as envisioned by its authors (i.e. posterior inference in the presence
of a Gaussian prior, which is used as a proposal distribution), and ESS applied
as a general purpose sampling method (i.e. by artificially introducing a 
Gaussian factor into the target density, using this Gaussian as a proposal 
distribution and the target density divided by it in place of the likelihood).
To emphasize that the latter viewpoint is the one deviating from the original
ESS paper, we refer to it as GP-ESS (short for "general purpose" and not to be
confused with "Gaussian prior").

The module provides the following single-chain sampling functions:
- ess: an implementation of ESS for posterior inference with Gaussian prior,
    taking as arguments the parameters of the prior (mean and covariance) and
    a functional representation of the likelihood
- gp_ess: a fully flexible implementation of GP-ESS, allowing for arbitrary
    choices of the artificial Gaussian factor (i.e. of its mean and covariance)
    and taking an arbitrary target density function as an argument
- naive_gp_ess: an optimized version of gp_ess in the case where the artificial
    Gaussian factor is chosen to be the multivariate standard Gaussian

Moreover, the module provides a parallelized version of naive_gp_ess with the
function parallel_ngp_ess.
"""

import numpy as np
import numpy.linalg as alg
import time as tm
from sampling_utils import nrange
from parallel_plain_sampling import parallel_plain
from threadpoolctl import threadpool_limits

def get_v_sampler(mean, cov, gen):
    """Auxiliary function, not to be called by the user"""
    d = mean.shape[0]
    if type(cov) == float or (type(cov) == np.ndarray and len(cov.shape) == 1):
        return ( lambda: np.sqrt(cov) * gen.normal(size=d) + mean )
    elif type(cov) == np.ndarray and len(cov.shape) == 2:
        chol_cov = alg.cholesky(cov)
        return ( lambda: chol_cov @ gen.normal(size=d) + mean )
    else:
        raise TypeError("Invalid type for argument cov! Consult the docstring" \
            + " regarding valid choices.")

def ellipse_shrinkage(log_like, mean, x_old, v, log_t, gen):
    """Auxiliary function, not to be called by the user"""
    ome = gen.uniform(0, 2*np.pi)
    ome_min, ome_max = ( ome - 2*np.pi, ome )
    x_prop = (x_old - mean) * np.cos(ome) + (v - mean) * np.sin(ome) + mean
    ldv_prop = log_like(x_prop)
    tde_cnt = 1
    while ldv_prop <= log_t:
        ome = gen.uniform(ome_min, ome_max)
        x_prop = np.cos(ome) * (x_old - mean) + np.sin(ome) * (v - mean) + mean
        ldv_prop = log_like(x_prop)
        tde_cnt += 1
        if ome < 0:
            ome_min = ome
        else:
            ome_max = ome
    return x_prop, ldv_prop, tde_cnt

@threadpool_limits.wrap(limits=1) # suppress numpy's automatic parallelization
def ess(mean, cov, log_like, n_its, x_0, gen, bar=True):
    """Runs ESS for posterior inference in presence of a Gaussian prior.

        Args:
            mean: mean parameter of the Gaussian prior, must be 1d np array
            cov: covariance parameter of the Gaussian prior. There are three
                different ways to set this argument (the first two are slightly
                more computationally optimized for their respective cases).
                The interpretation of cov as a representation of the prior's
                covariance matrix depends on the type of cov as follows:
                - float: covariance is cov * np.identity(d)
                - 1d np array: covariance is np.diag(cov)
                - 2d np array: covariance is cov itself
                Accordingly, in the second case cov must be of size d and in the
                third case of shape (d,d).
            log_like: log of the target density's likelihood factor, must
                be a function taking a size d np array as input and returning a
                float representing the value of the log likelihood at the given
                point
            n_its: number of iterations to run the algorithm for, must be non-
                negative integer
            x_0: initial state, must be size d np array describing a point 
                from the support of the target density
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure

        Returns:
            samples: np array of shape (n_its+1, d), where samples[i] is the 
                i-th sample generated by ESS
            gen: the generator given to the sampler in its latest state after
                the sampling performed in this call
            tde_cnts: 1d np array of size n_its+1, where tde_cnts[i] is the
                number of target density evaluations (in this case really target
                likelihood evaluations) used in the i-th iteration
            runtimes: 1d np array of size n_its+1, where runtimes[i] is the
                time the chain took to perform its i-th iteration (in seconds)
    """
    d = x_0.shape[0]
    sample_v = get_v_sampler(mean, cov, gen)
    X = np.zeros((n_its+1, d))
    X[0] = x_0
    ldv = np.zeros(n_its+1) # evals of log_like != log_density!
    ldv[0] = log_like(X[0])
    tde_cnts = np.zeros(n_its+1, dtype=int)
    tde_cnts[0] = 1
    runtimes = np.zeros(n_its+1)
    time_b = tm.time()
    for n in nrange(n_its, bar):
        v = sample_v()
        log_t = ldv[n-1] + np.log(gen.uniform())
        X[n], ldv[n], tde_cnts[n] \
            = ellipse_shrinkage(log_like, mean, X[n-1], v, log_t, gen)
        time_a = tm.time()
        runtimes[n] = time_a - time_b
        time_b = time_a
    return X, gen, tde_cnts, runtimes

@threadpool_limits.wrap(limits=1) # suppress numpy's automatic parallelization
def gp_ess(log_density, mean, cov, n_its, x_0, gen, bar=True):
    """Runs GP-ESS with a given (but a priori arbitrary) artificial Gaussian
        factor.

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            mean: mean parameter of the artificial Gaussian factor, must be 1d
                np array
            cov: covariance parameter of the artificial Gaussian factor. There
                are three different ways to set this argument (the first two are
                slightly more computationally optimized for their respective
                cases). The interpretation of cov as a representation of the
                Gaussian factor's covariance matrix depends on the type of cov
                as follows:
                - float: covariance is cov * np.identity(d)
                - 1d np array: covariance is np.diag(cov)
                - 2d np array: covariance is cov itself
                Accordingly, in the second case cov must be of size d and in the
                third case of shape (d,d).
            n_its: number of iterations to run the algorithm for, must be non-
                negative integer
            x_0: initial state, must be size d np array describing a point 
                from the support of the target density
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure

        Returns:
            samples: np array of shape (n_its+1, d), where samples[i] is the 
                i-th sample generated by ESS
            gen: the generator given to the sampler in its latest state after
                the sampling performed in this call
            tde_cnts: 1d np array of size n_its+1, where tde_cnts[i] is the
                number of target density evaluations used in the i-th iteration
            runtimes: 1d np array of size n_its+1, where runtimes[i] is the
                time the chain took to perform its i-th iteration (in seconds)
    """
    d = x_0.shape[0]
    sample_v = get_v_sampler(mean, cov, gen)
    if type(cov) == float or (type(cov) == np.ndarray and len(cov.shape) == 1):
        log_den_trf = lambda x: log_density(x) + np.inner(x, x / cov) / 2
    elif type(cov) == np.ndarray and len(cov.shape) == 2:
        inv_cov = alg.inv(cov)
        log_den_trf = lambda x: log_density(x) + np.inner(x, inv_cov @ x) / 2
    X = np.zeros((n_its+1, d))
    X[0] = x_0
    ldv = np.zeros(n_its+1) # evals of log_den_trf != log_density!
    ldv[0] = log_den_trf(X[0])
    tde_cnts = np.zeros(n_its+1, dtype=int)
    tde_cnts[0] = 1
    runtimes = np.zeros(n_its+1)
    time_b = tm.time()
    for n in nrange(n_its, bar):
        log_t = ldv[n-1] + np.log(gen.uniform())
        v = sample_v()
        X[n], ldv[n], tde_cnts[n] \
            = ellipse_shrinkage(log_den_trf, mean, X[n-1], v, log_t, gen)
        time_a = tm.time()
        runtimes[n] = time_a - time_b
        time_b = time_a
    return X, gen, tde_cnts, runtimes

@threadpool_limits.wrap(limits=1) # suppress numpy's automatic parallelization
def naive_gp_ess(log_density, n_its, x_0, gen, bar=False):
    """Runs naive GP-ESS (i.e. GP-ESS with the artificial Gaussian factor being
        chosen as the multivariate standard Gaussian), for a given number of
        iterations. This function is slightly more optimized for its particular
        setting than the corresponding call of gp_ess.

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_its: number of iterations to run the algorithm for, must be non-
                negative integer
            x_0: initial state, must be size d np array describing a point 
                from the support of the target density
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure

        Returns:
            samples: np array of shape (n_its+1, d), where samples[i] is the 
                i-th sample generated by naive GP-ESS
            gen: the generator given to the sampler in its latest state after
                the sampling performed in this call
            tde_cnts: 1d np array of size n_its+1, where tde_cnts[i] is the
                number of target density evaluations used in the i-th iteration
            runtimes: 1d np array of size n_its+1, where runtimes[i] is the
                time the chain took to perform its i-th iteration (in seconds)
    """
    d = x_0.shape[0]
    log_den_trf = lambda x: log_density(x) + np.sum(x**2) / 2
    X = np.zeros((n_its+1, d))
    X[0] = x_0
    ldv = np.zeros(n_its+1) # evals of log_den_trf != log_density!
    ldv[0] = log_den_trf(X[0])
    tde_cnts = np.zeros(n_its+1, dtype=int)
    tde_cnts[0] = 1
    runtimes = np.zeros(n_its+1)
    time_b = tm.time()
    for n in nrange(n_its, bar):
        log_t = ldv[n-1] + np.log(gen.uniform())
        v = gen.normal(size=d)
        # instead of using ellipse_shrinkage with mean = np.zeros(d),
        # re-implement the corresp. steps without mean for max efficiency:
        ome = gen.uniform(0, 2*np.pi)
        ome_min, ome_max = ( ome - 2*np.pi, ome )
        x_prop = np.cos(ome) * X[n-1] + np.sin(ome) * v
        ldv_prop = log_den_trf(x_prop)
        tde_cnts[n] = 1
        while log_den_trf(x_prop) <= log_t:
            ome = gen.uniform(ome_min, ome_max)
            x_prop = np.cos(ome) * X[n-1] + np.sin(ome) * v
            ldv_prop = log_den_trf(x_prop)
            tde_cnts[n] += 1
            if ome < 0:
                ome_min = ome
            else:
                ome_max = ome
        X[n] = x_prop
        ldv[n] = ldv_prop
        time_a = tm.time()
        runtimes[n] = time_a - time_b
        time_b = time_a
    return X, gen, tde_cnts, runtimes

def parallel_ngp_ess(
        log_density,
        n_chains,
        n_its,
        x_0s,
        verbose=True,
        bar=True
    ):
    """An implementation of naive GP-ESS that advances a number of chains in
        parallel (using CPU parallelization).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point; must be pickle-
                able (i.e. it mustn't be a lambda function)
            n_chains: number of parallel chains to use, must be positive integer
            n_its: number of iterations to perform per chain, must be positive 
                integer
            x_0s: initial states for the parallel chains, must be 2d np array of 
                shape (n_chains,d)
            w: initial width of the radius search area, must be positive float
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure (for practical reasons, the 
                progress bar will display the progression of an arbitrary chain,
                not that of the slowest one)

        Returns:
            samples: np array of shape (n_its+1,n_chains,d), where 
                samples[i,j] is the i-th sample generated by the j-th chain
            tde_cnts: 2d np array of shape (n_its+1,n_chains), where
                tde_cnts[i,j] is the number of target density evaluations used
                by the j-th chain in the i-th iteration
            runtimes: 2d np array of shape (n_its+1,n_chains), where
                runtimes[i,j] is the time the j-th chain took to perform its
                i-th iteration (in seconds)
    """
    return parallel_plain(
        naive_gp_ess,
        log_density,
        n_chains,
        n_its,
        x_0s,
        None,
        verbose,
        bar
    )

