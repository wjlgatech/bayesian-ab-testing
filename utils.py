# Utility functions for Bayesian AB Testing
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def plot_beta(alpha, beta, ax=None, title="", xlabel="",ylabel="", label=""):
    """plot the Beta distribution PDF with parameters alpha and beta
    Args
    ----
        alpha (positive number)
        beta (positive number)
        
    Example
    -------
    from ipywidgets import interactive, FloatSlider, IntSlider, fixed
    from IPython.display import display, Image

    fig, ax = plt.subplots()

    plot=interactive(plot_beta,
                     alpha=IntSlider(min=1,max=35,step=1,value=1),
                     beta=IntSlider(min=1,max=35,step=1,value=1),
                     ax=fixed(None), #fix other arguments
                     title=fixed(""), 
                     xlabel=fixed(""),
                     ylabel=fixed(""),
                     label=fixed("")
                    )
    display(plot)
    """
    # Build a beta distribtuion scipy object.
    dist = stats.beta(alpha, beta)

    # The support (always this for the beta dist).
    x = np.linspace(0.0, 1.0, 301)

    # The probability density at each sample support value.
    y = dist.pdf(x)

    # Plot it all.
    if ax is None:
        fig, ax = plt.subplots()
    xticks=[0.0, 0.5, 1.0]
    lines = ax.plot(x, y, label=label)
    ax.fill_between(x, y, alpha=0.2, color=lines[0].get_c())
    if title: 
        ax.set_title(title)
    else:
        ax.set_title(f'Beta distribution alpha={alpha}, beta={beta} ')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.get_yaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([np.max(y)])
    ax.get_xaxis().set_ticks(xticks)
    ax.set_ylim(0.0, min(np.max(y)*1.2,100))

def estimate_beta_params(data):
    """Estimate the alpha & beta parameters of beta distribution by fitting Beta distribution to the conversion data 
    Args
    ----
        data: a list of 0 (miss) or 1 (convert) or a 1-D np.array
    return
    ------
        Alpha: number of successes +1
        Beta: Number of failures +1
        Mean: conversion rate
        num_conversions & total_visitors: used to make labels on graphs 
    
    """
    #array of website conversions... zeros and ones (convert or didnt convert)
    website_samples = np.array(data)
    
    #total number of conversions
    num_conversions = website_samples.sum()
    #total number of datapoints
    total_visitors = len(website_samples)
    
    #plus one to set a and beta as uniform priors...try other numbers to see the changes
    alpha = num_conversions + 1
    beta = (total_visitors - num_conversions) + 1
    
    #mean number of conversions... aka conversion rate
    mean = 1 * num_conversions / total_visitors

    return alpha, beta, mean, num_conversions, total_visitors

def plot_beta_from_data(data, ax=None, label=None):
    """First estimate the Beta distribution parameters from data and then plot the Beta PDF distribution
    Args
    ----
        data: a list of 0 (miss) or 1 (convert)
    Examples
    --------
    plot_beta_from_data([0, 1, 0, 0, 0]*2)
    plot_beta_from_data([0, 1, 0, 0, 0]*20)
    plot_beta_from_data([0, 1, 0, 0, 0]*200)
    """
    alpha, beta, mean, num_conversions, total_visitors = estimate_beta_params(data)
    title =  r"Converted {}/{}".format(num_conversions, total_visitors)
    plot_beta(alpha, beta, ax=ax, title=title, xlabel="Conversion Rate", ylabel="Probability Density", label=label)

def compare_AB_conversion_rate(site_A_data, site_B_data, n=None):
    """compare the conversion rate of 2 different solution
    Args
    ----
        site_A_data (1-D np.array of shape (N,))
        site_B_data (1-D np.array of shape (N,))
        n (int): use a sub list of samples[:n]
    Return
    ------
        a plot comparing the conversion rate of site A and that of site B.
    
    Example
    -------
        interactive(compare_AB_conversion_rate, 
                n = IntSlider(min=1,max=len(x['site_A_samples']),step=10,value=1),
                site_A_data=fixed(x['site_A_samples']), 
                site_B_data=fixed(x['site_B_samples'])
               )
    """
    ax = plt.subplot()
    if n is None:
        n = min(len(site_A_data), len(site_A_data))
    site_A_samples = site_A_data[:n]
    site_B_samples = site_B_data[:n]
    plot_beta_from_data(site_A_samples, ax, label="Site A")
    plot_beta_from_data(site_B_samples, ax, label="Site B")
    plt.title(f'after {n} visitors')
    plt.legend()
    plt.show()

def compare_AB_by_simulations(site_A_samples, site_B_samples, n=None, num_simulations = 100_000):
    """ 1) show the probability that site B is better than site A
        2) create a `blob plot`: plotting random samples from B against random samples of A.  By measuring how much of the blob is above the y=x line, we can determine the probability that B is better than A.
    Args
    ----
        site_A_samples (1-D np.array) of 0 (miss) and 1 (convert)
        site_B_samples (1-D np.array) of 0 (miss) and 1 (convert)
        n (int): use a sub list of samples[:n]
    Return
    ------
        prob_B_betterthan_A: the probability of site B is better than A
    
    Example
    -------
        compare_AB_by_simulations(site_A_samples=x['site_A_samples'], site_B_samples=x['site_B_samples'])
    """
    if n is None:
        n = len(site_A_samples)

    #Let's just grab our Alpha and betas from site_A
    alpha_A, beta_A = estimate_beta_params(site_A_samples[:n])[:2]
    #print(f'Site_A alpha and beta {alpha, beta}')
    #Set up first distribution
    dist_A = stats.beta(alpha_A, beta_A)

    #Same steps for beta dist
    alpha_B, beta_B = estimate_beta_params(site_B_samples[:n])[:2]
    #print(f'Site_B alpha and beta {alpha, beta}')
    dist_B = stats.beta(alpha_B, beta_B)

    #randomly sample 100_000 data points from each distribution
    simulated_A = dist_A.rvs(num_simulations)
    simulated_B = dist_B.rvs(num_simulations)
    prob_B_betterthan_A = (simulated_B > simulated_A).mean()

    #print(f'On average, how many times is Bs Conversion Rate greater than As: {prob_B_betterthan_A}')
    
    #scatter plot our different conversion rates sampled from our distributions
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(simulated_A, simulated_B, alpha = .2);
    ax.set_xlim(plt.ylim())
    ax.set_xlabel(f'Site_A alpha and beta {alpha_A, beta_A}') 
    ax.set_ylabel(f'Site_B alpha and beta {alpha_B, beta_B}')
    ax.set_title(f'Using {n} samples, {prob_B_betterthan_A *100}% is Bs Conversion Rate greater than As')
    ax.plot(plt.xlim(), plt.xlim(), color = 'blue');
    return prob_B_betterthan_A


def bayes_credible_interval(site_x_samples, interval_size = 0.95, ax=None, title=''):
    """compute & plot bayesian credible interval
    Args
    ----
        site_x_samples (1-D np.array with shape (N,)): site-x's convertion data with 0 (miss) or 1 (convert),e.g. site_A_samples or site_B_samples
        inerval_size (float within [0,1], default .95): the size of credible interval
        
    Return
    ------
        credible_interval: a 2-tuple (lb, ub)
    Example
    -------
        fig, ax = plt.subplots(figsize=(10,10))    
        bayes_credible_interval(site_A_samples, interval_size = 0.95, ax=ax, title='Site-A')
        bayes_credible_interval(site_B_samples, interval_size = 0.95, ax=ax, title='Site-B')
    """

    alpha, beta = estimate_beta_params(site_x_samples)[:2]
    #print(f'Site_x alpha and beta {alpha, beta}')
    dist_x = stats.beta(alpha, beta)

    lb = (1-interval_size)/2 # i.e. 0.025 for interval_size = 0.95
    ub = 1 - lb # i.e. 0.975 for interval_size = 0.95
    
    x = np.linspace(*dist_x.ppf([.001, .999]),101)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(x, dist_x.pdf(x), label=title)
    ax.vlines(dist_x.ppf([lb, ub]), ymin = 0, ymax = dist_x.pdf(x).max(), linestyles='dotted')
    ax.fill_between(x, 0, dist_x.pdf(x), where = (x< dist_x.ppf(lb)) | (x > dist_x.ppf(ub) ));
    ax.set_xlabel("All possible values for conversion rate")
    ax.set_ylabel("PDF")
    ax.set_title(title+f" Conversion rate's {interval_size *100:0.1f}% credible interval  [{dist_x.ppf(lb):0.3f},{dist_x.ppf(ub):0.3f}]")
    ax.legend()
    return (dist_x.ppf(lb), dist_x.ppf(ub))


def plot_samplesize_belief_truth(site_x_samples, num_samples = [10, 100, 1000], ax=None, true_rate=None):
    """Show the number of samples needed to push your BELIEF to the TRUE conversion rate
    Args
    ----
        site_x_samples (1-D np.array of shape (N,)): site x's convertion data with 0 (miss) or 1 (convert)
        num_samples: a list of integers
        
    Example
    -------
        fig, ax = plt.subplots(figsize=(10,6))
        plot_samplesize_belief_truth(site_A_samples, num_samples = [10, 1000], ax=ax, true_rate=None)
        plot_samplesize_belief_truth(site_B_samples, num_samples = [10, 1000], ax=ax, true_rate=None)    
    """
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for k in num_samples:
        samples = site_x_samples[:k]
        plot_beta_from_data(samples, ax, label=f"After {k} samples")

    ax.set_title(f"Number of samples needed to push your BELIEF to the TRUE conversion rate {true_rate}")
    ax.legend();

