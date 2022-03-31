# -*- coding: utf-8 -*-
"""
Last update: 1 Feb 2022

@author: Maxime Clenet

This code display two scenarios:
- the circular law
- the circular law with a rank one perturbation.

"""

# Importation of the packages:

import numpy as np
import matplotlib.pyplot as plt


# %%

# The choice of the variables is similar to the article.
# We create the matrix B and display his spectrum.


def plot_spectrum(n=100, mu=0, alpha=1, title=True):
    """
    Plot the spectrum of the associated matrix.

    Parameters
    ----------
    N : int
        Dimension of the matrix. The default is 100.
    mu : float
        parameter corresponding to mu. The default is 0.
    alpha : float
        parameter corresponding to alpha. The default is 1.
    title : bool
        presence of the title on the plot. The default is True.

    Returns
    -------
    fig: figure of the spectrum.

    """

    const = (mu*alpha/np.sqrt(n))
    B = (np.random.randn(n, n)+const)*(1/(np.sqrt(n)*alpha))

    eig_B = np.linalg.eigvals(B)

    # The rest of the function is dedicated to the plot.

    radius = 1/alpha  # radius of the circle.

    abs_mu = mu  # perturbed value
    radius_mu = 0.1  # radius around the perturbation

    t = np.linspace(0, 2*np.pi, 100)

    fig = plt.figure(1, figsize=(10, 6))

    plt.plot(radius*np.cos(t), radius*np.sin(t), color='k', linewidth=3)
    if mu > 1/alpha:
        plt.plot(abs_mu+radius_mu*np.cos(t), radius_mu *
                 np.sin(t), linestyle='--', color='k', linewidth=3)

    plt.plot(eig_B.real, eig_B.imag, '.', color='k')
    plt.grid(color='lightgray', linestyle='--')
    plt.axis("equal")

    plt.xlabel(r"Real axis", fontsize=15)
    plt.ylabel("Imaginary axis", fontsize=15)
    if title:
        plt.title(r'Circular law, $\alpha$ = {}, $\mu$ = {} (N = {})'.format(
            alpha, mu, n), fontsize=15)
    plt.show()

    plt.close()

    return fig


# An exemple of the first scenario:
plot_spectrum(n=1000, mu=0, alpha=1, title=False)

# An exemple of the second scenario:
plot_spectrum(n=1000, mu=2, alpha=1, title=False)
