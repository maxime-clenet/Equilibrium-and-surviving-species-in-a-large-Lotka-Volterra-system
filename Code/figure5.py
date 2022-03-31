# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet


This file allows you to display the graph showing the
theoretical distribution vs. the empirical distribution.
"""

# Importation of the packages and required functions.

import numpy as np
import matplotlib.pyplot as plt
from lemkelcp import lemkelcp


# For more informations on this two functions see functions.py
from functions import density_abundance
from functions import final_function


# %%


def plot_distrib(alpha=2, mu=0.2, B_size=2000, law_type='normal'):
    """
    Return a  graph showing the difference
    between theoretical distribution vs. the empirical distribution
    of the abondances.

    Parameters
    ----------
    alpha : float in [sqrt(2),Infty), optional
        Associated to the alpha value in the paper.
        The default is 2.
    mu : float in (-infty,1], optional
        Associated to the mu value in the paper.
        The default is 0.2.
    B_size : int, optional
        Size of the square matrix B.
        The default is 2000.

    Returns
    -------
    fig : matplotlib.figure.

    """
    # Computations for the empirical distribution.
    # We find a solution using the pivot algorithm.
    if law_type == 'normal':
        const = (mu*alpha/np.sqrt(B_size))
        B = (np.random.randn(B_size, B_size)+const)*(1/(np.sqrt(B_size)*alpha))
    elif law_type == 'uniform':
        mu = 0
        B = (np.random.random((B_size, B_size))*2 *
             np.sqrt(3)-np.sqrt(3))*1/(np.sqrt(B_size)*alpha)

    q = np.ones(B_size)
    M = -np.eye(B_size)+B

    res_lcp = lemkelcp.lemkelcp(-M, -q, maxIter=10000)[0]
    res_lcp_pos = res_lcp[res_lcp != 0]

    (p, m, sigma) = final_function(alpha, mu)

    x = np.linspace(0.01, 4, 1000)

    fig = plt.figure(1, figsize=(10, 6))
    y = np.ones(len(x))
    for k, v in enumerate(x):
        y[k] = density_abundance(v, p, m, sigma, alpha, mu)

    plt.plot(x, y, linewidth=2.5, color='k')
    plt.hist(res_lcp_pos, density=True, bins=20,
             edgecolor='black', color='#0000003d')

    plt.xlabel('Abundances (x*)', fontsize=15)
    plt.ylabel('Density of the distribution (f)', fontsize=15)
    plt.xlim(0, 4)
    # plt.ylim(0,1.7)
    #plt.legend(loc='upper right')
    plt.show()

    return fig


plot_distrib(B_size=300, alpha=np.sqrt(3), mu=0, law_type='uniform')
