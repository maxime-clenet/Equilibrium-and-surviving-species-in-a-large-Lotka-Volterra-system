# -*- coding: utf-8 -*-
"""
@author: Maxime

Spectrum of the Hermitian random matrix B+B^T.
"""


import numpy as np
import matplotlib.pyplot as plt


# %%

# The choice of the variables is similar to the article.
# We create the matrix B and display the spectrum of B+B^T.


def plot_capitaine_spectrum(n=100, mu=0, alpha=1, title=True):
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

    eig_B = np.linalg.eigvals(B+B.T)

    # The rest of the function is dedicated to the plot.

    t = np.linspace(-2*np.sqrt(2/alpha**2), 2*np.sqrt(2/alpha**2), 100)

    fig = plt.figure(1, figsize=(10, 6))

    if mu > 1/(np.sqrt(2)*alpha):
        plt.vlines(2*mu+1/(mu*alpha**2), 0, 0.35,
                   linestyles='--', color='k', linewidth=2)
    plt.plot(t, np.sqrt(4*(2/(alpha**2))-t**2) /
             (2*np.pi*(2/alpha**2)), color='k', linewidth=3)
    plt.hist(eig_B.real, density=True, bins=40,
             edgecolor='black', color='#0000003d')
    plt.grid(color='lightgray', linestyle='--')

    plt.xlim(-2, 3.5)
    plt.ylim(0, 0.35)

    plt.xlabel(r"Spectrum", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    if title:
        plt.title(r'Semi-circular law, $\alpha$ = {}, $\mu$ = {} (N = {})'.format(
            alpha, mu, n), fontsize=15)
    plt.show()

    plt.close()

    return fig
# plt.savefig('Spectre_Ginibre_Presentation_Lille.pdf')


# An exemple of the first scenario:
plot_capitaine_spectrum(n=1000, mu=0, alpha=np.sqrt(2), title=False)

# An exemple of the second scenario:
plot_capitaine_spectrum(n=1000, mu=1.5, alpha=np.sqrt(2), title=False)
