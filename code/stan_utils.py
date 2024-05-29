"""
Provides a convenient wrapper for the PyStan interface to the Stan sampling
library. In particular, this module replaces the broken parallelization of
Pystan itself by 'manual' parallelization.
"""

import numpy as np
import numpy.random as rnd
import nest_asyncio
nest_asyncio.apply()
import asyncio as asy
from concurrent.futures import ThreadPoolExecutor, as_completed
import stan

# utility string for Bayesian logistic regression model building
BLR_CODE = """
data {
    // actual data:
    int n_data;
    int d;
    matrix[n_data,d] a;
    array[n_data] int b;
    // hyperparameter:
    real sigma;
}
parameters {
    vector[d] x;
}
model {
    x ~ normal(0, sigma);
    b ~ bernoulli_logit_glm(a, zeros_vector(n_data), x);
}
"""


# this function is called for each model to be used by sample_in_parallel in
# accordance with the recommendation from https://discourse.mc-stan.org/t/
# trouble-with-pystan-3-and-python-multiprocessing/23846/6
def prevent_crashes(code, data):
    """Constructs a Stan posterior to cache and runs it for a single iteration
        to build trust and prevent the parallel sampling from crashing (no one
        seems to know why this is necessary, see https://discourse.mc-stan.org/
        t/trouble-with-pystan-3-and-python-multiprocessing/23846/6
    """
    # compile the Stan program implementing the posterior
    posterior = stan.build(code, data)
    posterior.sample(num_chains=1, num_samples=1, num_warmup=0)

def blr_data(n_data, d, a, b, sigma):
    """Constructs a Stan data dictionary for Bayesian logistic regression with
        isotropic Gaussian prior
    """
    data = {
        'n_data': n_data,
        'd': d,
        'a': a,
        'b': (b+1)//2,
        'sigma': sigma,
    }
    return data


# the parallelization is implemented following https://discourse.mc-stan.org/t/
# trouble-with-pystan-3-and-python-multiprocessing/23846/6

def get_fit_thread(code, data, seed, num_samples, init):
    """Utility function, not to be called by the user"""
    # create event loop for the thread
    loop = asy.new_event_loop()
    asy.set_event_loop(loop)
    # build and fit
    posterior = stan.build(code, data, seed)
    fit = posterior.sample(num_chains=1, num_samples=num_samples, init=init)
    return fit

def sample_in_parallel(code, data, n_chains, n_its, x_0s):
    """Samples in parallel from a given Stan posterior and processes the outputs
        to some extent before returning them.
    """
    d = x_0s.shape[-1]
    seeds = rnd.randint(1, int(1e6), size=n_chains)
    with ThreadPoolExecutor(max_workers=n_chains) as executor:
        futures = {}
        for i in range(n_chains):
            futures[(executor.submit(
                get_fit_thread,
                code = code,
                data = data,
                seed = int(seeds[i]),
                num_samples = n_its,
                init = [{'x': x_0s[i]}],
            ))] = i
        fits = [future.result() for future in as_completed(futures)]
    frames = [fit.to_frame() for fit in fits]
    samples = np.concatenate([frame.to_numpy()[:,-d:].reshape(n_its,1,d) \
                              for frame in frames], axis=1)
    tge = np.array([fit['n_leapfrog__'].reshape(-1) for fit in fits]).T
    acc_rates = np.array([fit['accept_stat__'].reshape(-1) for fit in fits]).T
    return frames, samples, tge, acc_rates

