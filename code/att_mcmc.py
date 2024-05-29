"""
Implements affine transformation tuning (ATT) of MCMC while enabling update
schedules. The module provides a general template for single-chain ATT by the
function

    att_mcmc

and sampler-specific versions of it by the functions

    att_gpss
    att_hruss
    att_rsuss
    att_ess

Moreover, the module provides naively parallelized versions of all the above
by the functions

    npatt_mcmc
    npatt_gpss
    npatt_hruss
    npatt_rsuss
    npatt_ess
"""

import numpy as np
import numpy.random as rnd
import numpy.linalg as alg
import multiprocessing as mp
import time as tm
from tqdm import tqdm # tqdm.notebook.tqdm would look nicer but is incompatible
# with the multiprocessing module

from affine_transformations import affine_trf, inv_affine_trf
from gibbsian_polar_slice_sampling import gpss
from hit_and_run_uniform_slice_sampling import hruss
from random_scan_uniform_slice_sampling import rsuss
from elliptical_slice_sampling import naive_gp_ess

def att_mcmc(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        sampler=None,
        hyper_burn=None,
        hyper_att=None,
        gen=rnd.default_rng(),
        bar=True,
        verbose=True,
    ):
    """Provides a general template for affine transformation tuning of MCMC,
        with the base sampler, the types of transformation parameter choices
        and the update schedule all being free to choose. Naive parallelization
        is enabled by passing an RNG to the function.

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0: initial value for the chain, must be np array of size d
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            sampler: underlying base sampler, must take as arguments (in this
                order!)
                - log density
                - number of iterations to perform
                - current state
                - possibly a hyperparameter
                - an instance of np.random.Generator 
                and must return
                - the samples it generated (2d np array)
                - the RNG in its latest state
                - the target density evaluation counts (1d np array)
                - the iteration-wise runtimes (1d np array)
                If this argument is set to None, Gibbsian polar slice sampling 
                will be used as the default sampler. If no value for its 
                hyperparameter is provided, the parameter will be set to one.
            hyper_burn: hyperparameter(s) of the base sampler to be used during
                initialization burn-in period, leave as None if there aren't any
            hyper_att: hyperparameter(s) of the base sampler to be used during
                ATT period, leave as None if there aren't any
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if verbose:
        print("Checking validity of given arguments...")
    if type(n_burn) != int or n_burn < 0:
        raise TypeError("Number of burn-in iterations must be non-negative" \
            + "integer!")
    if type(n_its) != int or n_its <= 0:
        raise TypeError("Number of iterations must be positive integer!")
    if type(x_0) != np.ndarray or len(x_0.shape) != 1:
        raise TypeError("Initial value must be given as 1d np array of size d!")
    if cen_mode not in [None, "mean", "medi"]:
        raise TypeError("Invalid value of cen_mode! Valid choices are None, " \
            + "\"mean\", and \"medi\".")
    if cov_mode not in [None, "var", "cov"]:
        raise TypeError("Invalid value of cov_mode! Valid choices are None, " \
            + "\"var\", and \"cov\".")
    auto_schedule = (type(schedule) == type(None))
    if not auto_schedule:
        if type(schedule) != np.ndarray or schedule.dtype != int or \
            (schedule.shape[0] > 1 and np.min(schedule[1:]-schedule[:-1]) <= 0)\
            or schedule[0] <= 1 or schedule[-1] >= n_its:
            raise TypeError("Schedule must be np array containing a strictly " \
                  + "increasing sequence of integers larger one and smaller " \
                  + "than n_its!")
    if verbose:
        print("Preparing for sampling...")
    d = x_0.shape[0]
    if sampler == None:
        sampler = gpss
        if hyper_burn == None:
            hyper_burn = 1
        if hyper_att == None:
            hyper_att = 1
    # if appropriate, run init burn-in:
    if n_burn > 0:
        if verbose:
            print("Starting init burn-in sampling...")
        if hyper_burn == None:
            ret = sampler(log_density, n_burn, x_0, gen)
        else:
            ret = sampler(log_density, n_burn, x_0, hyper_burn, gen)
        X_b = ret[0]
        gen = ret[1]
        tde_cnts_burn = ret[2]
        runtimes_burn = ret[3]
    if verbose:
        print("Preparing for ATT sampling...")
    # if no schedule was given, construct default schedule
    if auto_schedule:
        if cen_mode == "medi":
            raw_sched = 1.5**np.arange(17,np.log(n_its-1)/np.log(1.5))
            schedule = np.array(raw_sched, dtype=int)
        elif cov_mode == "cov":
            schedule = np.arange(10*max(d,25), n_its-1, 10*max(d,25))
        else:
            schedule = np.arange(250, n_its-1, 250)
    # construct augmented schedule
    s = np.concatenate([np.array([1]), schedule, np.array([n_its+1])])
    n_updates = s.shape[0] # technically no. of updates is this -1
    # prepare sample and tuning parameter storage (index the tuning parameters
    # via k rather than s_k) and correctly initialize recursively used arrays
    X = np.zeros((n_its+1, d))
    if n_burn > 0:
        X[0] = X_b[-1]
    else:
        X[0] = x_0
    if cen_mode == "mean" or cov_mode != None:
        m = np.zeros((n_updates,d))
        m[0] = X[0]
    if cen_mode == "medi":
        z = np.zeros((n_updates,d))
    if cov_mode == "var":
        q = np.zeros((n_updates,d))
        dev = np.zeros((n_updates,d))
    elif cov_mode == "cov":
        Q = np.zeros((n_updates,d,d))
        Sig = np.zeros((n_updates,d,d))
        L = np.zeros((n_updates,d,d))
        L_inv = np.zeros((n_updates,d,d))
    cen_para = None
    cov_para = None
    inv_cov_para = None
    alpha = affine_trf(cen_para, cov_para)
    alpha_inv = inv_affine_trf(cen_para, inv_cov_para)
    log_den_trd = lambda y: log_density(alpha(y))
    tde_cnts = np.zeros(n_its+1, dtype=int)
    runtimes = np.zeros(n_its+1)
    if verbose:
        print("Starting ATT sampling...")
    # create progress bar
    if bar:
        tqdm._instances.clear()
        pb = tqdm(total=n_its+1, position=0, leave=True)
        pb.update(1)
    # run the sampling procedure
    for k in range(1, n_updates):
        time_b = tm.time()
        y_old = alpha_inv(X[s[k-1]-1])
        time_a = tm.time()
        runtimes[s[k-1]] += time_a - time_b
        if hyper_att == None:
            ret = sampler(log_den_trd, s[k]-s[k-1], y_old, gen)
        else:
            ret = sampler(log_den_trd, s[k]-s[k-1], y_old, hyper_att, gen)
        time_b = tm.time()
        Y = ret[0][1:]
        X[s[k-1]:s[k]] = np.apply_along_axis(alpha, -1, Y)
        gen = ret[1]
        tde_cnts[s[k-1]:s[k]] = ret[2][1:]
        runtimes[s[k-1]:s[k]] += ret[3][1:]
        # update transformation params
        if cen_mode == "mean" or cov_mode != None:
            m[k] = m[k-1] + np.sum(X[s[k-1]:s[k]] - m[k-1], axis=0) / s[k]
            if cen_mode == "mean":
                cen_para = m[k]
        if cen_mode == "medi":
            # note that this does not actually use a linear time algo yet
            z[k] = np.median(X[:s[k]], axis=0)
            cen_para = z[k]
        if cov_mode == "var":
            q[k] = q[k-1] + np.sum( (X[s[k-1]:s[k]] - m[k-1]) \
                * (X[s[k-1]:s[k]] - m[k]), axis=0)
            dev[k] = np.sqrt(q[k] / (s[k]-1))
            cov_para = dev[k]
            inv_cov_para = 1 / dev[k]
        elif cov_mode == "cov":
            Q[k] = Q[k-1] + np.einsum("ij,ik->jk", X[s[k-1]:s[k]] - m[k-1], \
                X[s[k-1]:s[k]] - m[k])
            Sig[k] = Q[k] / (s[k]-1)
            L[k] = alg.cholesky(Sig[k])
            L_inv[k] = alg.inv(L[k])
            cov_para = L[k]
            inv_cov_para = L_inv[k]
        alpha = affine_trf(cen_para, cov_para)
        alpha_inv = inv_affine_trf(cen_para, inv_cov_para)
        log_den_trd = lambda y: log_density(alpha(y))
        time_a = tm.time()
        runtimes[s[k]-1] += time_a - time_b
        if bar:
            pb.update(s[k]-s[k-1])
    if verbose:
        print("Assembling output...")
    ret_dic = {}
    if n_burn > 0:
        ret_dic['burn-in'] = X_b
        ret_dic['tde_cnts_burn'] = tde_cnts_burn
        ret_dic['runtimes_burn'] = runtimes_burn
    ret_dic['samples'] = X
    ret_dic['tde_cnts'] = tde_cnts
    ret_dic['runtimes'] = runtimes
    if cen_mode == "mean" or cov_mode != None:
        ret_dic['means'] = m
    if cen_mode == "medi":
        ret_dic['medians'] = z    
    if cov_mode == "var":
        ret_dic['std_devs'] = dev
    if cov_mode == "cov":
        ret_dic['covs'] = Sig
    if auto_schedule:
        ret_dic['schedule'] = s
    return ret_dic

def att_gpss(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        w_burn=None,
        w_att=None,
        gen=rnd.default_rng(),
        bar=True,
        verbose=True,
    ):
    """Implements affine transformation tuning (ATT) of Gibbsian polar slice
        sampling (GPSS).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0: initial value for the chain, must be np array of size d
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            w_burn: value of the interval width hyperparameter of GPSS that is
                to be used during the initialization burn-in period, must be 
                positive float
            w_att: value of the interval width hyperparameter of GPSS that is to
                be used during the ATT period, must be positive float
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if n_burn > 0:
        if type(w_burn) not in [int, float, np.float64] or w_burn <= 0:
            raise TypeError("Hyperparameter w_burn must be a positive float!")
    if type(w_att) not in [int, float, np.float64] or w_att <= 0:
        raise TypeError("Hyperparameter w_att must be a positive float!")
    return att_mcmc(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode,
        cov_mode,
        schedule,
        gpss,
        w_burn,
        w_att,
        gen,
        bar,
        verbose,
    )

def att_hruss(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        w_burn=None,
        w_att=None,
        gen=rnd.default_rng(),
        bar=True,
        verbose=True,
    ):
    """Implements affine transformation tuning (ATT) of hit-and-run uniform
        slice sampling (HRUSS).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0: initial value for the chain, must be np array of size d
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            w_burn: value of the interval width hyperparameter of HRUSS that is
                to be used during the initialization burn-in period, must be 
                positive float
            w_att: value of the interval width hyperparameter of HRUSS that is
                to be used during the ATT period, must be positive float
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if n_burn > 0:
        if type(w_burn) not in [int, float, np.float64] or w_burn <= 0:
            raise TypeError("Hyperparameter w_burn must be a positive float!")
    if type(w_att) not in [int, float, np.float64] or w_att <= 0:
        raise TypeError("Hyperparameter w_att must be a positive float!")
    return att_mcmc(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode,
        cov_mode,
        schedule,
        hruss,
        w_burn,
        w_att,
        gen,
        bar,
        verbose,
    )

def att_rsuss(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        w_burn=None,
        w_att=None,
        gen=rnd.default_rng(),
        bar=True,
        verbose=True,
    ):
    """Implements affine transformation tuning (ATT) of random scan uniform
        slice sampling (RSUSS).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0: initial value for the chain, must be np array of size d
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            w_burn: value of the interval width hyperparameter of RSUSS that is
                to be used during the initialization burn-in period, must be 
                positive float
            w_att: value of the interval width hyperparameter of RSUSS that is
                to be used during the ATT period, must be positive float
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if n_burn > 0:
        if type(w_burn) not in [int, float, np.float64] or w_burn <= 0:
            raise TypeError("Hyperparameter w_burn must be a positive float!")
    if type(w_att) not in [int, float, np.float64] or w_att <= 0:
        raise TypeError("Hyperparameter w_att must be a positive float!")
    return att_mcmc(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode,
        cov_mode,
        schedule,
        rsuss,
        w_burn,
        w_att,
        gen,
        bar,
        verbose,
    )

def att_ess(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        gen=rnd.default_rng(),
        bar=True,
        verbose=True,
    ):
    """Implements affine transformation tuning (ATT) of elliptical slice
        sampling (ESS).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0: initial value for the chain, must be np array of size d
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation during sampling
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    return att_mcmc(
        log_density,
        n_burn,
        n_its,
        x_0,
        cen_mode,
        cov_mode,
        schedule,
        naive_gp_ess,
        None,
        None,
        gen,
        bar,
        verbose,
    )

def npatt_mcmc(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        sampler=None,
        hyper_burn=None,
        hyper_att=None,
        bar=True,
        verbose=True,
    ):
    """Provides a general template for naively parallelized affine transforma-
        tion tuning (NPATT) of MCMC, with the underlying base sampler, the
        types of transformation parameter choices and the update schedule all
        being free to choose.

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_chains: number of parallel chains to use, must be positive integer
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0s: initial values for the parallel chains, must be 2d np array of 
                shape (n_chains,d)
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            sampler: underlying base sampler, must take as arguments (in this
                order!)
                - log density
                - number of iterations to perform
                - current state
                - possibly a hyperparameter
                - an instance of np.random.Generator 
                and must return
                - the samples it generated (2d np array)
                - the RNG in its latest state
                - the target density evaluation counts (1d np array)
                - the iteration-wise runtimes (1d np array)
                If this argument is set to None, Gibbsian polar slice sampling 
                will be used as the default sampler. If no value for its 
                hyperparameter is provided, the parameter will be set to one.
            hyper_burn: hyperparameter(s) of the base sampler to be used during
                initialization burn-in period, leave as None if there aren't any
            hyper_att: hyperparameter(s) of the base sampler to be used during
                ATT period, leave as None if there aren't any
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure (for practical reasons, the 
                progress bar will display the progression of an arbitrary chain,
                not that of the slowest one)
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if type(n_chains) != int or n_chains <= 0:
        raise TypeError("Number of parallel chains must be positive integer!")
    # initialize generators for random numbers
    seeds = rnd.SeedSequence().spawn(n_chains)
    gens = [rnd.Generator(rnd.MT19937(s)) for s in seeds]
    # assemble arguments for parallel call
    if verbose:
        print("Preparing for parallel sampling...")
    bars = [bar] + (n_chains - 1) * [False]
    starmap_args = [(log_density, n_burn, n_its, x_0, cen_mode, cov_mode,
        schedule, sampler, hyper_burn, hyper_att, gen, b, False)
        for x_0, gen, b in zip(x_0s, gens, bars)]
    # run parallel chains and process the returns
    if verbose:
        print("Starting parallel sampling...")
    pool = mp.Pool(n_chains)
    returns = pool.starmap(att_mcmc, starmap_args)
    pool.close()
    if verbose:
        print("Processing returns and assembling output...")
    ret_dic = {}
    if n_burn > 0:
        ret_dic['burn-in'] = np.transpose(np.array([ret['burn-in']
            for ret in returns]), axes=(1,0,2))
        ret_dic['tde_cnts_burn'] = np.array([ret['tde_cnts_burn']
            for ret in returns]).T
        ret_dic['runtimes_burn'] = np.array([ret['runtimes_burn']
            for ret in returns]).T
    ret_dic['samples'] = np.transpose(np.array([ret['samples']
        for ret in returns]), axes=(1,0,2))
    ret_dic['tde_cnts'] = np.array([ret['tde_cnts'] for ret in returns]).T
    ret_dic['runtimes'] = np.array([ret['runtimes'] for ret in returns]).T
    if cen_mode == "mean" or cov_mode != None:
        ret_dic['means'] = np.array([ret['means'] for ret in returns])
    if cen_mode == "medi":
        ret_dic['medians'] = np.array([ret['medians'] for ret in returns])
    if cov_mode == "var":
        ret_dic['std_devs'] = np.array([ret['std_devs'] for ret in returns])
    if cov_mode == "cov":
        ret_dic['covs'] = np.array([ret['covs'] for ret in returns])
    if type(schedule) == type(None):
        ret_dic['schedule'] = returns[0]['schedule']
    return ret_dic

def npatt_gpss(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        w_burn=None,
        w_att=None,
        bar=True,
        verbose=True,
    ):
    """Implements naively parallelized affine transformation tuning (NPATT) of
        Gibbsian polar slice sampling (GPSS).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_chains: number of parallel chains to use, must be positive integer
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0s: initial values for the parallel chains, must be 2d np array of 
                shape (n_chains,d)
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            w_burn: value of the interval width hyperparameter of GPSS that is
                to be used during the initialization burn-in period, must be 
                positive float
            w_att: value of the interval width hyperparameter of GPSS that is to
                be used during the ATT period, must be positive float
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure (for practical reasons, the 
                progress bar will display the progression of an arbitrary chain,
                not that of the slowest one)
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if n_burn > 0:
        if type(w_burn) not in [int, float, np.float64] or w_burn <= 0:
            raise TypeError("Hyperparameter w_burn must be a positive float!")
    if type(w_att) not in [int, float, np.float64] or w_att <= 0:
        raise TypeError("Hyperparameter w_att must be a positive float!")
    return npatt_mcmc(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode,
        cov_mode,
        schedule,
        gpss,
        w_burn,
        w_att,
        bar,
        verbose,
    )

def npatt_hruss(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        w_burn=None,
        w_att=None,
        bar=True,
        verbose=True,
    ):
    """Implements naively parallelized affine transformation tuning (NPATT) of
        hit-and-run uniform slice sampling (HRUSS).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_chains: number of parallel chains to use, must be positive integer
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0s: initial values for the parallel chains, must be 2d np array of 
                shape (n_chains,d)
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            w_burn: value of the interval width hyperparameter of HRUSS that is
                to be used during the initialization burn-in period, must be 
                positive float
            w_att: value of the interval width hyperparameter of HRUSS that is to
                be used during the ATT period, must be positive float
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure (for practical reasons, the 
                progress bar will display the progression of an arbitrary chain,
                not that of the slowest one)
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if n_burn > 0:
        if type(w_burn) not in [int, float, np.float64] or w_burn <= 0:
            raise TypeError("Hyperparameter w_burn must be a positive float!")
    if type(w_att) not in [int, float, np.float64] or w_att <= 0:
        raise TypeError("Hyperparameter w_att must be a positive float!")
    return npatt_mcmc(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode,
        cov_mode,
        schedule,
        hruss,
        w_burn,
        w_att,
        bar,
        verbose,
    )

def npatt_rsuss(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        w_burn=None,
        w_att=None,
        bar=True,
        verbose=True,
    ):
    """Implements naively parallelized affine transformation tuning (NPATT) of
        random scan uniform slice sampling (RSUSS).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_chains: number of parallel chains to use, must be positive integer
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0s: initial values for the parallel chains, must be 2d np array of 
                shape (n_chains,d)
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            w_burn: value of the interval width hyperparameter of RSUSS that is
                to be used during the initialization burn-in period, must be 
                positive float
            w_att: value of the interval width hyperparameter of RSUSS that is to
                be used during the ATT period, must be positive float
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure (for practical reasons, the 
                progress bar will display the progression of an arbitrary chain,
                not that of the slowest one)
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    if n_burn > 0:
        if type(w_burn) not in [int, float, np.float64] or w_burn <= 0:
            raise TypeError("Hyperparameter w_burn must be a positive float!")
    if type(w_att) not in [int, float, np.float64] or w_att <= 0:
        raise TypeError("Hyperparameter w_att must be a positive float!")
    return npatt_mcmc(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode,
        cov_mode,
        schedule,
        rsuss,
        w_burn,
        w_att,
        bar,
        verbose,
    )

def npatt_ess(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode=None,
        cov_mode=None,
        schedule=None,
        bar=True,
        verbose=True,
    ):
    """Implements naively parallelized affine transformation tuning (NPATT) of
        elliptical slice sampling (ESS).

        Args:
            log_density: log of the target density, must be a function taking
                a size d np array as input and returning a float representing
                the value of the log density at the given point
            n_chains: number of parallel chains to use, must be positive integer
            n_burn: number of iterations to perform in initialization burn-in 
                period (which consists of sampling without ATT to reach states 
                that are a suitable initialization for ATT), must be non-neg.
                integer
            n_its: number of iterations to perform after the initialization 
                burn-in, must be positive integer
            x_0s: initial values for the parallel chains, must be 2d np array of 
                shape (n_chains,d)
            cen_mode: centering mode for tuning, valid options are None, "mean",
                and "medi" (corresponding to no adjustment, mean-centering and
                median-centering, respectively)
            cov_mode: covariance adjustment mode for tuning, valid options are
                None, "var", and "cov" (corresponding to no adjustment, variance
                adjustment and covariance adjustment, respectively)
            schedule: update schedule, must be None or a np array containing a 
                strictly increasing sequence of integers larger one, the largest
                of which must be smaller than n_its. If left as None, a
                schedule S = (s_k)_{k=1,2,...} will be constructed based on 
                cen_mode and cov_mode as follows:
                - mean, var, mean+var: s_k = 250*k
                - cov, mean+cov: s_k = 10*max(d,25)*k
                - medi, medi+var, medi+cov: s_k = floor(1.5**(k+16))
                For the user's convenience, an automatically generated schedule 
                will be added to the return dictionary.
            bar: bool denoting whether or not a progress bar should be displayed
                during the sampling procedure (for practical reasons, the 
                progress bar will display the progression of an arbitrary chain,
                not that of the slowest one)
            verbose: bool denoting whether or not to print various status
                updates during the sampling procedure
    """
    return npatt_mcmc(
        log_density,
        n_chains,
        n_burn,
        n_its,
        x_0s,
        cen_mode,
        cov_mode,
        schedule,
        naive_gp_ess,
        None,
        None,
        bar,
        verbose,
    )

