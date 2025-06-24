"""
Implements generalized elliptical slice sampling (GESS), as proposed in the
paper "Parallel MCMC with Generalized Elliptical Slice Sampling" by Nishihara,
Murray and Adams (2014). The module implements GESS both as a single-chain
sampler using a fixed approximation of the target distribution by a multivariate
t-distribution, and as a two-group parallel sampler that updates the groups 
alternately, while computing the approximation for one group from the states of
the other group.
"""

import numpy as np
import numpy.linalg as alg
import numpy.random as rnd
import scipy.special as sps
import scipy.optimize as opt
import multiprocessing as mp
from tqdm.notebook import tqdm
from elliptical_slice_sampling import ellipse_shrinkage
from sampling_utils import nrange
from threadpoolctl import threadpool_limits

# loosely based on github.com/robertnishihara/gess/blob/master/fit_mvstud.py
def fit_multivariate_t(data, min_df=1e-1, max_df=1e6, tol=1e-6, max_its=500):
    """Auxiliary function, not to be called by the user"""
    (n,d) = data.shape
    # initialize the t-dist's parameters (the initial value for df being a wild
    # guess is okay because we update it first below)
    df = 20
    center = np.median(data, axis=0)
    scale = np.cov(data, rowvar=False, bias=True) + 0.1 * np.identity(d)
    # iteratively optimize them
    df_old = 0
    its = 0
    cen_data = data - center
    while np.abs(df - df_old) > tol and its < max_its:
        # compute relative distances and update df
        deltas = np.sum(cen_data * alg.solve(scale, cen_data.T).T, axis=1)
        df_old = df
        def equation(df_):
            ws_ = (df_ + d) / (df_ + deltas)
            return -sps.psi(df_/2) + np.log(df_/2) + np.mean(np.log(ws_) - ws_)\
                + sps.psi((df_ + d)/2) - np.log((df_ + d)/2) + 1
        if equation(min_df) < 0:
            return min_df, center, scale
        if equation(max_df) >= 0:
            return max_df, center, scale
        df = opt.brentq(equation, min_df, max_df)
        # compute weights and update center and scale
        ws = (df + d) / (df + deltas.reshape(n,1))
        center = np.sum(ws * data, axis=0) / np.sum(ws)
        cen_data = data - center
        scale = (ws * cen_data).T @ cen_data / n
        its += 1
    return df, center, scale

@threadpool_limits.wrap(limits=1) # suppress numpy's automatic parallelization
def generalized_ess(
        log_density,
        df,
        center,
        scale,
        n_its,
        x_0,
        gen,
        bar=False
    ):
    """Runs generalized elliptical slice sampling, as proposed by Nishihara et
        al. (2014), as a single-chain method with a fixed approximation of the
        target by a multivariate t-distribution.

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            df: degrees of freedom parameter of the multivariate t-distribution
                used to approximate the target, must be positive float
            center: center parameter of the multivariate t-distribution used to
                approximate the target, must be 1d np array
            scale: scale matrix parameter of the multivariate t-distribution
                used to approximate the target. There are three different ways
                to set this argument (the first two are slightly more computa-
                tionally optimized for their respective cases). The interpreta-
                tion of scale as a representation of the approximation's scale
                matrix depends on the type of scale as follows:
                - float: scale matrix is scale * np.identity(d)
                - 1d np array: scale matrix is np.diag(scale)
                - 2d np array: scale matrix is scale itself
                Accordingly, in the second case scale must be of size d and in
                the third case of shape (d,d).
            n_its: number of iterations to run the algorithm for, must be non-
                negative integer
            x_0: initial state, must be size d np array describing a point 
                from the support of the target density
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure
    """
    d = x_0.shape[0]
    if type(scale)==float or (type(scale)==np.ndarray and len(scale.shape)==1):
        quad_form = lambda x: np.inner(x - center, (x - center) / scale)
        scalify = lambda x: np.sqrt(scale) * x
    elif type(scale) == np.ndarray and len(scale.shape) == 2:
        inv_scale = alg.inv(scale)
        quad_form = lambda x: np.inner(x - center, inv_scale @ (x - center))
        chol_scale = alg.cholesky(scale)
        scalify = lambda x: chol_scale @ x
    else:
        raise TypeError("Invalid type for argument scale! Consult the" \
            + " docstring regarding valid choices.")
    a = (d + df)/2
    log_den_trf = lambda x: log_density(x) + a * np.log(1 + quad_form(x)/df)
    X = np.zeros((n_its+1,d))
    X[0] = x_0
    ldv = np.zeros(n_its+1) # evals of log_den_trf != log_density!
    ldv[0] = log_den_trf(X[0])
    tde_cnts = np.zeros(n_its+1, dtype=int)
    tde_cnts[0] = 1
    for n in nrange(n_its, bar):
        log_t = ldv[n-1] + np.log(gen.uniform())
        b = (df + quad_form(X[n-1]))/2
        s = 1/gen.gamma(a, 1/b)
        v = np.sqrt(s) * scalify(gen.normal(size=d)) + center
        X[n], ldv[n], tde_cnts[n] \
        = ellipse_shrinkage(log_den_trf, center, X[n-1], v, log_t, gen)
    return X, gen, tde_cnts

def two_group_gess(
        log_density,
        n_chains,
        n_its,
        x_0s,
        n_ibu=None,
        n_thr=None,
        bar=True,
        verbose=True
    ):
    """Runs the two-group parallel version of generalized elliptical slice
        sampling, as proposed by Nishihara et al. (2014). In particular, this
        function follows the authors' suggestion to re-use the approximation
        parameters for a (fixed) number of iterations before updating them.

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_chains: number of parallel chains to use in each group, must be
                positive integer
            n_its: number of iterations to run the algorithm for, must be non-
                negative integer (more precisely, in order to yield the same
                total number of samples as other samplers of similar call
                signature, the chains of each group will only be run for n_its/2
                iterations each)
            x_0s: initial values for the parallel chains of both groups, must be
                2d np array of shape (2*n_chains,d), where the first n_chains
                rows represent the initial values for the first group and so on
            n_ibu: number of iterations between (consecutive) updates of the
                parameters of the multivariate t-distribution approximation to
                the target distribution, must be positive integer; if left None,
                it will be chosen as n_ibu = max(d,25) * n_chains / 10
            n_thr: number of parallel threads to use, must be positive integer;
                if left None, n_chains parallel threads will be used
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if verbose:
        print("Checking validity of given arguments...")
    if type(n_chains) != int or n_chains <= 0:
        raise TypeError("Number of parallel chains must be positive integer!")
    if type(n_its) != int or n_its <= 0:
        raise TypeError("Number of iterations must be positive integer!")
    if type(x_0s) != np.ndarray or len(x_0s.shape) != 2 \
        or x_0s.shape[0] != 2 * n_chains:
        raise TypeError("Initial values must be given as 2d np array of shape" \
            + " (2*n_chains,d)!")
    if n_ibu != None and (type(n_ibu) != int or n_ibu <= 0):
        raise TypeError("Number of iterations between updates must be positive"\
            + " integer!")
    d = x_0s.shape[-1]
    if n_chains < d:
        print("WARNING: The sampler is likely to run into numerical issues if" \
            + " the number of chains in each group is smaller than the sample" \
            + " space dimension!")
    if verbose:
        print("Preparing for parallel sampling...")
    # prepare update schedule
    n_ipc = n_its // 2 # iterations per chain
    if n_ibu == None:
        n_ibu = max(d,25) * n_chains // 10
    schedule = np.arange(max(n_ibu,2), n_ipc-1, n_ibu)
    s = np.concatenate([np.array([1]), schedule, np.array([n_ipc+1])])
    n_updates = s.shape[0] # technically no. of updates is this -1
    # prepare sample and approximation parameter storage
    X_G1 = np.zeros((n_ipc+1, n_chains, d))
    X_G2 = np.zeros((n_ipc+1, n_chains, d))
    X_G1[0] = x_0s[:n_chains]
    X_G2[0] = x_0s[n_chains:]
    tde_cnts_G1 = np.zeros((n_ipc+1, n_chains), dtype=int)
    tde_cnts_G2 = np.zeros((n_ipc+1, n_chains), dtype=int)
    df_G1 = np.zeros(n_updates)
    df_G2 = np.zeros(n_updates)
    cen_G1 = np.zeros((n_updates, d))
    cen_G2 = np.zeros((n_updates, d))
    scale_G1 = np.zeros((d,d))
    scale_G2 = np.zeros((d,d))
    # prepare RNGs
    seeds = rnd.SeedSequence().spawn(n_chains)
    gens = [rnd.Generator(rnd.MT19937(s)) for s in seeds]
    if verbose:
        print("Starting two-group sampling...")
    # create progress bar
    if bar:
        tqdm._instances.clear()
        pb = tqdm(total=2*(n_ipc+1), position=0, leave=True)
        pb.update(2)
    # initialize process pool
    if n_thr != None:
        pool = mp.Pool(n_thr)
    else:
        pool = mp.Pool(n_chains)
    # run the sampling procedure
    for k in range(1, n_updates):
        # update group 1
        df_G1[k], cen_G1[k], scale_G1 = fit_multivariate_t(X_G2[s[k-1]-1])
        starmap_args = [(log_density, df_G1[k], cen_G1[k], scale_G1, \
            s[k]-s[k-1], x, gen) for x, gen in zip(X_G1[s[k-1]-1], gens)]
        returns = pool.starmap(generalized_ess, starmap_args)
        new_samples = np.array([ret[0][1:] for ret in returns])
        X_G1[s[k-1]:s[k]] = np.transpose(new_samples, axes=(1,0,2))
        gens = [ret[1] for ret in returns]
        tde_cnts_G1[s[k-1]:s[k]] = np.array([ret[2][1:] for ret in returns]).T
        if bar:
            pb.update(s[k]-s[k-1])
        # update group 2
        df_G2[k], cen_G2[k], scale_G2 = fit_multivariate_t(X_G1[s[k]-1])
        starmap_args = [(log_density, df_G2[k], cen_G2[k], scale_G2, \
            s[k]-s[k-1], x, gen) for x, gen in zip(X_G2[s[k-1]-1], gens)]
        returns = pool.starmap(generalized_ess, starmap_args)
        new_samples = np.array([ret[0][1:] for ret in returns])
        X_G2[s[k-1]:s[k]] = np.transpose(new_samples, axes=(1,0,2))
        gens = [ret[1] for ret in returns]
        tde_cnts_G2[s[k-1]:s[k]] = np.array([ret[2][1:] for ret in returns]).T
        if bar:
            pb.update(s[k]-s[k-1])
    pool.close()
    # assemble return dictionary and terminate
    if verbose:
        print("Assembling output...")
    ret_dic = {}
    ret_dic['samples'] = np.concatenate([X_G1, X_G2], axis=1)
    ret_dic['tde_cnts'] = np.concatenate([tde_cnts_G1, tde_cnts_G2], axis=1)
    ret_dic['dfs'] = np.array([df_G1, df_G2])
    ret_dic['centers'] = np.array([cen_G1, cen_G2])
    ret_dic['scales'] = np.array([scale_G1, scale_G2])
    ret_dic['schedule'] = s
    return ret_dic

