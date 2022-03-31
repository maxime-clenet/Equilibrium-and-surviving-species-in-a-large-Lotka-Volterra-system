# -*- coding: utf-8 -*-
"""
@author: Maxime

This program returns curves on m,sigma and p the proportion of persistent species.
A 3D plot representing sigma,m or p in function of (mu,alpha)


The execution of this program relies exclusively on the "final_function".
See Function.py for more information.
"""


# Importation of the packages and required functions.

from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from functions import final_function

# %%

# This part concerned a 3D plot representing sigma in function of (mu,alpha)

# =========================
# generating ordered data:


def properties(bound_alpha=(1.5, 2.5), bound_mu=(-0.5, 0.5), prec=20, prop_type=2):
    """
    3D plot representing a choosen properties in function of (alpha,mu)

    Parameters
    ----------
    bound_alpha : tuple, optional
        1st value corresponds to lower bound of alpha,
        the 2nd to upper bound. Interval in [sqrt(2),infty].
        The default is (1.5,2.5)
    bound_mu : tuple, optional
        1st value corresponds to lower bound of mu,
        the 2nd to upper bound. Interval in [-1,1].
    prec : int, optional
        Accuracy of the graph grid. The default is 20.
    prop_type : 0,1 or 2, optional
        0 - return p* in function of (alpha,mu),
        1 - return m* in function of (alpha,mu),
        2 - return sigma* in function of (alpha,mu).
        The default is 1.

    Returns
    -------
    fig_prop : matplotlib.figure
        3D representation.

    """
    # generating ordered data:
    x = np.linspace(bound_alpha[0], bound_alpha[1], prec)

    y = np.linspace(bound_mu[0], bound_mu[1], prec)

    X, Y = np.meshgrid(x, y, indexing='ij')

    Z = np.zeros((prec, prec))

    for k in range(prec):
        for j in range(prec):
            Z[j, k] = final_function(X[j, k], Y[j, k])[prop_type]
    # reference picture (X, Y and Z in 2D):

    fig_prop = plt.figure()
    ax = fig_prop.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z,
                           cmap='Greys', linewidth=0)
    fig_prop.colorbar(surf)

    #title = ax.set_title("Feasibility phase diagram")
    # title.set_y(1.01)
    plt.xlabel(r"Interaction strength ($ \alpha $)")
    plt.ylabel(r"Interaction drift ($\mu$)")
    #plt.zlabel(r"$ \pi $")

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    #ax.view_init(40, -110)

    fig_prop.tight_layout()

    return fig_prop


properties(prec=100, prop_type=2)
