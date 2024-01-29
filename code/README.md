### Overview of Code Base and Experiments

This directory contains all the source code belonging to the paper.

The experiments presented in the paper, as well as two that were cut for space,
are each organized as a single Jupyter notebook. To look at them, it should be
sufficient to have the essential Jupyter packages installed. If one is interes-
ted in running them, the file

    requirements.txt

lists all the python packages required to do so. To have the packages at the
versions stated therein is likely not strictly necessary, the listed versions
are merely the ones we happened to have installed.

We now give an overview of the source files and notebooks provided in this
directory, starting with the latter.

Our main experiments, i.e. all those summarized in Section 5 and described in
detail in Appendix G, are implemented by the following notebooks:

    Bayesian_inference_with_multivariate_exponential_distributions.ipynb
    BLR_German_credit_data.ipynb
    BLR_breast_cancer_data.ipynb
    BLR-FE_Pima_diabetes_data.ipynb
    BLR-FE_wine_quality_data.ipynb
    Bayesian_hyperparameter_inference_for_GP_regression_census_data.ipynb

The ablation studies of Appendix H are implemented by these notebooks:

    ablation_study_adjustment_types.ipynb
    ablation_study_parallelization_and_update_schedules.ipynb
    ablation_study_init_burn-in.ipynb

In the following notebooks, we conducted simple experiments on toy targets with
two further slice samplers (other than ESS and GPSS) to demonstrate how PATT,
applied suitably, can improve their performance (for which we simply compared
each method's PATT sampler with a naively parallelized non-PATT version). The
numbers in the titles represent a suggest reading order, but the notebooks
should also be understandable on their own.

    performance_gain_1_HRUSS.ipynb
    performance_gain_2_RSUSS.ipynb

There are also some simple testing scripts to ensure all the samplers and a
utils module are in proper working order:

    testing_plain_samplers.ipynb
    testing_gess_samplers.ipynb
    testing_att_samplers.ipynb
    testing_patt_samplers.ipynb
    testing_patt_adjustment_types.ipynb
    testing_mcmc_utils.ipynb

The "backend" of all of these notebooks is implemented by a number of different
modules. For starters, the following implement the base samplers as single-chain
methods, naively parallelized versions of them, as well as the methods AdaRWM
and GESS that were used as competitors for PATT in the paper:

    sampling_utils.py
    standard_sampling_functions_gen.py
    parallel_plain_sampling.py
    gibbsian_polar_slice_sampling.py
    hit_and_run_uniform_slice_sampling.py
    random_scan_uniform_slice_sampling.py
    elliptical_slice_sampling.py
    generalized_elliptical_slice_sampling.py
    random_walk_metropolis.py

Some more modules are required to implement ATT (single-chain and naively
parallelized) and PATT:

    affine_transformations.py
    att_mcmc.py
    patt_mcmc.py

Finally, we provide two utils modules meant to ease analysis of the output of
MCMC-style samplers through the computation of performance metrics and the
generation of plots:

    mcmc_utils.py
    plotting_functions.py

