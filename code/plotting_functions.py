"""
Provides various MCMC-related plotting functions with many options and 
reasonable defaults wherever possible

Available functions:
    plot_2d_samples
    plot_3d_samples
    plot_samples
    contour_precalc
    plot_contour_and_samples
    trace_plot
    trace_plot_row
    plot_tde_distr
    plot_tde_distr_row
    plot_step_hist
    plot_step_hist_row
    plot_ada_progress
    plot_ada_progress_testing
    plot_adaptation
    plot_covs_row
    plot_trace_and_step_hists
    plot_traces_2_col
    plot_trace_steps_tde
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import mcmc_utils as mcu

################################################################################
###################### Auxiliary Variables and Functions #######################

markers = ["o", "s", "^", "v", "P", "d"]
nmarkers = len(markers)

def initiate(figsize, dpi, title=None):
    """Auxiliary function, not to be called by the user"""
    plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    if title != None:
        plt.title(title)

def initiate_overview(figsize, dpi, nsub):
    """Auxiliary function, not to be called by the user"""
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    subfigs = fig.subfigures(nrows=nsub, ncols=1)
    return subfigs

def wrapup(filepath=None):
    """Auxiliary function, not to be called by the user"""
    if filepath != None:
        plt.savefig(filepath)
    plt.show()

def size_gen(size, n):
    """Auxiliary function, not to be called by the user"""
    if size != None:
        return size
    return np.min([np.max([1e3/n, 0.025]), 1.0])

def bin_gen(nbins, nvals):
    """Auxiliary function, not to be called by the user"""
    if nbins != None:
        return nbins
    return np.max([nvals//1000, 1])

################################################################################
################################ User Functions ################################

################ Plotting of 2d and 3d Samples and 2d Contours #################

def plot_2d_samples(
        samples, 
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None
    ):
    """Creates a simple scatter plot of 2-dimensional samples

        Args:
            samples: np array of shape (nsamples,2)
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by scatter(), by default the
                size used is 1e3/nsamples projected to the interval [0.025, 1.0]
    """
    initiate(figsize, dpi, title)
    size = size_gen(size, samples.shape[0])
    plt.scatter(samples[:,0], samples[:,1], s=size)
    wrapup(filepath)

def plot_3d_samples(
        samples,
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None
    ):
    """Creates a simple scatter plot of 3-dimensional samples

        Args:
            samples: np array of shape (nsamples,3)
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by scatter(), by default the
                size used is 1e3/nsamples projected to the interval [0.025, 1.0]
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    size = size_gen(size, samples.shape[0])
    ax = fig.add_subplot(projection="3d")
    if title != None:
        ax.set_title(title)
    ax.scatter(samples[:,0], samples[:,1], samples[:,2], s=size)
    wrapup(filepath)

def plot_samples(
        samples,
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None
    ):
    """Creates a simple scatter plot of 2- or 3-dimensional samples, simply
        terminates if sample dimension is not 2 or 3

        Args:
            samples: np array of shape (nsamples,d) with d in [2,3]
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by scatter(), by default the
                size used is 1e3/nsamples projected to the interval [0.025, 1.0]
    """
    d = samples.shape[1]
    if d == 2:
        plot_2d_samples(samples, figsize, dpi, title, filepath, size)
    elif d == 3:
        plot_3d_samples(samples, figsize, dpi, title, filepath, size)

def contour_precalc(
        reso,
        x1min,
        x1max,
        x2min,
        x2max,
        fct
    ):
    """Computes quantities required for contour and 3d plots

        Args:
            reso: resolution (total number of grid points) in each
                coordinate direction
            x1min: minimal value along x1-axis
            x1max: maximal value along x1-axis
            x2min: minimal value along x2-axis
            x2max: maximal value along x2-axis
            fct: function to be evaluated, should take 1d np arrays
                of length 2 as input
        Returns:
            G1, G2, vals: arguments for plt.contour() etc.
    """
    x1s = np.linspace(x1min, x1max, reso)
    x2s = np.linspace(x2min, x2max, reso)
    G1, G2 = np.meshgrid(x1s, x2s)
    X = np.concatenate([G1.reshape(reso,reso,1),G2.reshape(reso,reso,1)],axis=2)
    vals = np.zeros(G1.shape)
    for i in range(reso):
        vals[i] = np.array(list(map(fct, X[i])))
    return G1, G2, vals

def plot_contour_and_samples(
        G1,
        G2,
        vals,
        samples,
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None,
        levels=8,
        filled=False
    ):
    """Plots contours of a bivariate target density and given (typically 
        approximate) samples from it in the same figure

        Args:
            G1, G2: grid values for contour plot, like produced by np.meshgrid
            vals: function values of the function to be contour plotted at the 
                grid locations given by G1, G2
            samples: np array of shape (nsamples,2)
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by plt.scatter, by default the size 
                used is 1e3/nsamples projected to the interval [0.025, 1.0]
            levels: number of levels or actual levels for contour plot
            filled: whether space between contour lines should be filled
                with (approximately) continuous contours
    """
    initiate(figsize, dpi, title)
    contfct = plt.contourf if filled else plt.contour
    contfct(G1, G2, vals, levels)
    size = size_gen(size, samples.shape[0])
    plt.scatter(samples[:,0], samples[:,1], s=size, color="red")
    wrapup(filepath)

################################# Trace Plots ##################################

def trace_plot(
        vals,
        figsize=(3,2),
        dpi=100,
        title=None,
        filepath=None,
        lw=None
    ):
    """Creates a trace plot of the given values

        Args:
            vals: values to be plotted, should be 1d np array
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            lw: linewidth to be used by plt.plot(), by default the width used
                is 1e3/vals.shape[0] projected to the interval [0.025, 1.0]
    """
    initiate(figsize, dpi, title)
    nvals = vals.shape[0]
    lw = size_gen(lw, nvals)
    plt.plot(range(nvals), vals, linewidth=lw)
    wrapup(filepath)

def trace_plot_row(
        vals,
        snames,
        spsize=(3,2),
        dpi=100,
        title=None,
        filepath=None,
        lws=None
    ):
    """Creates a row of trace plots of given values

        Args:
            vals: values to be plotted, should be 1d np array
            snames: names of the samplers used to be printed in legend
            spsize: 2-tuple giving the size of each subplot
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            lws: linewidth or linewidths to be used by plt.plot(), by default
                the width used is in column i is 1e3/vals[i].shape[0] projected
                to the interval [0.025, 1.0]
    """
    nsam = len(snames)
    if type(lws) in [type(None), int, float]:
        lws = nsam * [lws]
    figsize = (nsam * spsize[0], spsize[1])
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=nsam)
    if title != None:
        plt.title(title)
    for i in range(nsam):
        axes[i].set_title(snames[i])
        lw = size_gen(lws[i], vals[i].shape[0])
        axes[i].plot(vals[i], linewidth=lw)
    wrapup(filepath)

############################ Statistical Quantities ############################

def plot_tde_distr(
        tde_cnts,
        figsize=(3,2),
        dpi=100,
        title=None,
        filepath=None
    ):
    """Cumulates counts of target density evaluations, presents the cumulated
        counts in a bar plot.
        
        Args:
            tde_cnts: the TDE counts, must be np array (of arbitrary shape) 
                containing non-negative ints
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
    """
    evals, cum_cnts = np.unique(tde_cnts, return_counts=True)
    initiate(figsize, dpi, title)
    plt.bar(evals, cum_cnts / np.sum(cum_cnts), width=0.5)
    wrapup(filepath)

def plot_tde_distr_row(
        tde_cnts,
        snames,
        spsize=(3,2),
        dpi=100,
        title=None,
        filepath=None,
    ):
    """Cumulates counts of target density evaluations, presents the cumulated
        counts in a row of bar plots, one for each sampler
        
        Args:
            tde_cnts: the TDE counts, must be np array (of arbitrary shape) 
                containing non-negative ints
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it

        Args:
            tde_cnts: the TDE counts, must be a list of np arrays (of
                arbitrary shape) containing non-negative ints
            snames: names of the samplers used to be printed in legend
            spsize: 2-tuple giving the size of each subplot
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
    """
    nsam = len(snames)
    figsize = (nsam * spsize[0], spsize[1])
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=nsam)
    if title != None:
        plt.title(title)
    for i in range(nsam):
        axes[i].set_title(snames[i])
        evals, cum_cnts = np.unique(tde_cnts[i], return_counts=True)
        axes[i].bar(evals, cum_cnts / np.sum(cum_cnts), width=0.5)
    wrapup(filepath)

def plot_step_hist(
        samples,
        figsize=(3,2),
        dpi=100,
        title=None,
        filepath=None,
        nbins=None
    ):
    """Plots a histogram of the distances between consecutive samples

        Args:
            samples: np array of shape (nsamples,d) containing the samples
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            nbins: number of bins to be used, by default the number will be set
                so that the average bin contains 1000 elements
    """
    steps = alg.norm(samples[1:] - samples[:-1], axis=1)
    nbins = bin_gen(nbins, steps.shape[0])
    initiate(figsize, dpi, title)
    plt.hist(steps, bins=nbins)
    wrapup(filepath)

def plot_step_hist_row(
        steps,
        snames,
        spsize=(3,2),
        dpi=100,
        title=None,
        filepath=None,
        nbins=None,
        cutoff_quant=0.995,
        same_range=False,
    ):
    """Creates a row of step histogram plots corresponding to given step sizes

        Args:
            steps: list of arrays of step sizes for each sampler
            snames: names of the samplers used to be printed in legend
            spsize: 2-tuple giving the size of each subplot
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            nbins: number of bins to be used, by default the number will be set
                so that the average bin contains 1000 elements
            cutoff_quant: quantile after which to cut off the histogram
            same_range: whether to restrict all histograms to the same range
    """
    nsam = len(snames)
    figsize = (nsam * spsize[0], spsize[1])
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=nsam)
    if title != None:
        plt.title(title)
    if same_range:
        maxstep = nsam * [ np.max([np.quantile(stps, cutoff_quant) \
                                   for stps in steps]) ]
    else:
        maxstep = [np.quantile(stps, cutoff_quant) for stps in steps]
    nbins = bin_gen(nbins, steps[0].shape[0])
    for i in range(nsam):
        axes[i].set_title(snames[i])
        axes[i].hist(steps[i], bins=nbins, range=(0,maxstep[i]))
        axes[i].set_xlim(0,maxstep[i])
    wrapup(filepath)

################################ Miscellaneous #################################

def plot_ada_progress(
        schedule,
        params,
        gt,
        figsize=(3,2),
        dpi=250,
        title=None,
        filepath=None,
        marker=None,
    ):
    """Plot PATT's adaptation progress, i.e. the progression of the distance (in
        log-scale) between its tuning parameter and the underlying ground truth
        value that parameter aims to approximate. The distance underlying the
        presented values is the Euclidean distance between the (flattened, if
        necessary) parameter and ground truth vectors.

        Args:
            schedule: the update schedule from which the given tuning parameters
                resulted, must be 1d np array of size n_updates
            params: the progression of values of a tuning parameter (e.g. the
                center), must be np array satisfying params.shape[0] = n_updates
            gt: ground truth value of the parameter approximated by params, must
                be np array of the same shape as params[0]
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            marker: symbol to use for plt.plot's marker-argument, leave None
                if no such markers are desired
    """
    n_ups = schedule.shape[0]
    params = params.reshape(n_ups, -1)
    gt = gt.reshape(-1)
    devs = alg.norm(params - gt, axis=1)
    initiate(figsize, dpi, title)
    plt.yscale("log")
    plt.plot(schedule, devs, marker=marker)
    wrapup(filepath)

def plot_ada_progress_testing(
        schedule,
        params,
        gt,
        pname,
        figsize=(3,2),
        dpi=100,
        filepath=None,
        marker=None,
    ):
    """Wrapper for plot_ada_progress with fixed title template and smaller
        default dpi, meant for use in testing notebooks.
    """
    title = "{} Adaptation Error".format(pname)
    plot_ada_progress(
        schedule,
        params,
        gt,
        figsize,
        dpi,
        title,
        filepath,
        marker
    )

def plot_adaptation(
        schedule,
        vals,
        figsize=(3,2),
        dpi=100,
        title=None,
        filepath=None,
        lw=None,
        marker=None,
    ):
    """Visualize (P)ATT's adaptation through a trace plot of a summary statistic
        (usually a norm) of the parameters that are being adapted.

        Args:
            schedule: the update schedule from which the given tuning parameters
                resulted, must be 1d np array of size n_updates
            vals: the progression of summary statistic values of a tuning para-
                meter (e.g. the center), must be 1d np array of size n_updates
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            lw: linewidth to be used by plt.plot(), by default the width used
                is 1e3/vals.shape[0] projected to the interval [0.025, 1.0]
            marker: symbol to use for plt.plot's marker-argument, leave None
                if no such markers are desired
    """
    initiate(figsize, dpi, title)
    plt.plot(schedule, vals, marker=marker)
    wrapup(filepath)

def plot_covs_row(
        covs,
        titles,
        spsize=(3,3),
        dpi=250,
        filepath=None,
        shrink=0.75,
    ):
    """Plots several covariance matrices next to each other in a row, each with
        its own colorbar.
    
        Args:
            covs: list of 2d np arrays of shape (d,d)
            titles: list of titles for the matrices in covs
            spsize: 2-tuple giving the size of each subplot
            dpi: dots per inch used for plotting
            filepath: location to save plot to, leave default to not save it
            shrink: factor by which to shrink the colorbar (there seems to be no
                universally good value to make plt.imshow() and plt.colorbar()
                align in their y-extents
    """
    ncov = len(covs)
    initiate((ncov*spsize[0],spsize[1]), dpi)
    for i, title, cov in zip(range(ncov), titles, covs):
        plt.subplot(1,ncov,i+1)
        plt.title(title)
        plt.imshow(cov)
        plt.colorbar(shrink=shrink)
        plt.xticks([])
        plt.yticks([])
    wrapup(filepath)

################################ Overview Plots ################################

def plot_trace_and_step_hists(
        vals,
        steps,
        snames,
        figsize=(10,10),
        dpi=100,
        title=None,
        filepath=None,
        lws=None,
        nbins=None,
        cutoff_quant=0.995,
        same_range=True,
    ):
    """Creates a plot that contains (nsam,2) subplots, with each row
        containing a trace plot and a step size histogram for one algorithm

        Args:
            vals: list of size nsam containing the 1d quantities for each 
                sampler that should be trace plotted
            steps: list of size nsam containing the step sizes for each sampler
            snames: list of length nsam containing names of the samplers used 
                to be printed as row titles
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            lws: linewidth or linewidths to be used by plt.plot(), by default
                the width used is in column i is 1e3/vals[i].shape[0] projected
                to the interval [0.025, 1.0]
            nbins: number of bins to be used, by default the number will be set
                so that the average bin contains 1000 elements
            cutoff_quant: quantile of step sizes at which to cut off the
                histogram
            same_range: bool denoting whether to restrict all histograms to the
                same x-range
    """
    nsam = len(snames)
    if type(lws) in [type(None), int, float]:
        lws = nsam * [lws]
    if same_range:
        maxstep = nsam * [ np.max([np.quantile(stps, cutoff_quant) \
                                   for stps in steps]) ]
    else:
        maxstep = [np.quantile(stps, cutoff_quant) for stps in steps]
    subfigs = initiate_overview(figsize, dpi, nsam)
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(snames[i])
        axes = subfig.subplots(nrows=1, ncols=2)
        # left column: trace plot
        lw = size_gen(lws[i], vals[i].shape[0])
        axes[0].plot(vals[i], linewidth=lw)
        # right column: step size histogram
        nb = bin_gen(nbins, steps[i].shape[0])
        axes[1].hist(steps[i], bins=nb, range=(0,maxstep[i]))
        axes[1].set_xlim(0,maxstep[i])
        axes[1].yaxis.tick_right()
    wrapup(filepath)

def plot_traces_2_col(
        vals1,
        vals2,
        snames,
        figsize=(10,10),
        dpi=100,
        filepath=None,
        lw1=None,
        lw2=None
    ):
    """Creates a plot that contains (nsam,2) subplots, with each row
        containing two trace plot for one algorithm

        Args:
            vals1: list of size nalg containing 1d np arrays to be trace-plotted
            vals2: list of size nalg containing 1d np arrays to be trace-plotted
            snames: list of length nsam containing names of the samplers used 
                to be printed as row titles
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            filepath: location to save plot to, leave default to not save it
            lw1: linewidth to be used in left column of trace plots, by default 
                the width used in row i is 1e3/vals1[i].shape[0] projected to
                the interval [0.025, 1.0]
            lw2: linewidth to be used in right column of trace plots, by default
                the width used in row i is 1e3/vals2[i].shape[0] projected to
                the interval [0.025, 1.0]
    """
    subfigs = initiate_overview(figsize, dpi, len(vals1))
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(snames[i])
        axes = subfig.subplots(nrows=1, ncols=2)
        axes[0].plot(vals1[i], linewidth=lw1)
        axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        axes[1].plot(vals2[i], linewidth=lw2)
        axes[1].yaxis.tick_right()
    wrapup(filepath)

def plot_trace_steps_tde(
        vals,
        steps,
        tde,
        snames,  
        spsize=(4,2),
        dpi=200,
        filepath=None,
        lws=None,
        nbins=None,
        cutoff_quant=0.995,
        same_range=True,
    ):
    """Creates a plot that contains (3,nsam) subplots, with each column contai-
        ning a trace plot, a step size histogram and a TDE distribution plot for
        one method's samples

        Args:
            vals: list of length nsam containing 1d np arrays with the values to
                be trace plotted in the plot's first row
            steps: list of length nsam containing 1d np arrays with all the step
                sizes of which a histogram is to be displayed in the plot's
                second row
            tde: list of length nsam containing 1d or 2d np arrays with the TDE
                counts of which a bar plot is to be displayed in the plot's
                third row
            snames: list of length nsam containing names of the samplers used to
                be printed as row titles
            spsize: 2-tuple giving the figsize to be used for each subplot
            dpi: dots per inch used for plotting
            filepath: location to save plot to, leave default to not save it
            lws: linewidth or linewidths to be used by plt.plot(), by default
                the width used is in column i is 1e3/vals[i].shape[0] projected
                to the interval [0.025, 1.0]
            nbins: number of bins to be used, by default the number will be set
                so that the average bin contains 1000 elements
            cutoff_quant: quantile of step sizes at which to cut off the
                histogram
            same_range: whether to restrict both histograms to the same range
    """
    nsam = len(snames)
    if type(lws) in [type(None), int, float]:
        lws = nsam * [lws]
    if same_range:
        maxstep = nsam * [ np.max([np.quantile(stps, cutoff_quant) \
                                   for stps in steps]) ]
    else:
        maxstep = [np.quantile(stps, cutoff_quant) for stps in steps]
    initiate((nsam*spsize[0],3*spsize[1]), dpi)
    for i in range(nsam):
        # first row: a trace plot
        ax = plt.subplot(3,nsam,i+1)
        ax.set_title(snames[i])
        lw = size_gen(lws[i], vals[i].shape[0])
        ax.plot(vals[i], linewidth=lw)
        # second row: step size histograms
        ax = plt.subplot(3,nsam,nsam+i+1)
        nb = bin_gen(nbins, steps[i].shape[0])
        ax.hist(steps[i], bins=nb, range=(0,maxstep[i]))
        ax.set_xlim(0,maxstep[i])
        # third row: TDE distributions
        ax = plt.subplot(3,nsam,2*nsam+i+1)
        evals, cum_cnts = np.unique(tde[i][1:], return_counts=True)
        ax.bar(evals, cum_cnts / np.sum(cum_cnts), width=0.5)
        ax.set_xticks(np.arange(5,np.max(evals)+1,5))
    for j in [nsam, 2*nsam, 3*nsam]:
        ax = plt.subplot(3,nsam,j)
        ax.yaxis.tick_right()
    for i in range(2,nsam):
        for j in range(3):
            ax = plt.subplot(3,nsam,j*nsam+i)
            ax.set_yticks([])
    wrapup(filepath)

