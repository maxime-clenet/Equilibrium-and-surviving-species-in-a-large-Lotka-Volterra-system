# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

This program returns curves on m,sigma and p the proportion of persistent species.
The file is divided in two categories:
- plot of p,m or sigma in function of alpha.
- plot of p,m or sigma in function of mu.

The main goal is to compare the theoretical solution with the empirical ones.


The execution of this program relies exclusively on:
- the "final_function" for the theoretical part,
- the "empirical_prop" for the empirical part.
See Function.py for more information.
"""


# Importation of the packages and required functions.

import numpy as np
import matplotlib.pyplot as plt
from functions import final_function, empirical_prop

# %%

# Display p (the proportion of persistent species) in function of alpha.


# The first part is dedicated to the theoretical solution.


PREC_ALPHA = 20  # precision of the plot.
MU = 0  # fix the parameter mu.

# We choose the interval of parameter to execute the function.
init_alpha = np.linspace(1.4, 3.5, PREC_ALPHA)

# Storage of the result:
theo_sol = np.ones((PREC_ALPHA, 3))


for i in range(PREC_ALPHA):
    try:
        alpha = init_alpha[i]
        theo_sol[i, :] = final_function(alpha, MU)
    except ValueError:
        theo_sol[i] = 0
        print("problem")


# The second part is dedicated to the empirical solution:

B_size = 500

emp_sol = np.zeros((PREC_ALPHA, 3))

for i in range(PREC_ALPHA):
    alpha = init_alpha[i]
    print("Iteration:", i+1, '/', PREC_ALPHA)

    emp_sol[i, :] = empirical_prop(
        B_size=B_size, alpha=alpha, mu=MU, mc_prec=200)


# The following is dedicated to the plot.
# For more information, see functions.py
# Remark: theo_sol[:, 0] -> p
# theo_sol[:, 1] -> m
# theo_sol[:, 2] -> sigma

def plot_alpha(title):
    if title == 'p':
        fig = plt.figure(1, figsize=(10, 6))
        plt.plot(init_alpha, theo_sol[:, 0], 'k', label=r'$p$')
        plt.plot(init_alpha, emp_sol[:, 0], 'k*', label='$\widehat{p}$')
        plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
        plt.ylabel(r"Proportion of the surviving species ($p)$", fontsize=15)
        plt.show()
    elif title == 'm':
        fig = plt.figure(1, figsize=(10, 6))
        plt.plot(init_alpha, theo_sol[:, 1], 'k', label=r'$m$')
        plt.plot(init_alpha, emp_sol[:, 1], 'k*', label='$\widehat{m}$')
        plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
        plt.ylabel(r"Mean of the surviving species ($m*)$", fontsize=15)
        plt.show()
    elif title == 'sigma':
        fig = plt.figure(1, figsize=(10, 6))
        plt.plot(init_alpha, theo_sol[:, 2], 'k', label=r'$\sigma$')
        plt.plot(init_alpha, emp_sol[:, 2], 'k*', label='$\widehat{\sigma}$')
        plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
        plt.ylabel(
            r"Root mean square of the surviving species ($\sigma*)$", fontsize=15)
        plt.show()
    else:
        fig = 'Choose p,m or sigma'

    return fig


plot_alpha('p')
plot_alpha('m')
plot_alpha('sigma')


# %%

# Display p (the proportion of persistent species) in function of mu.
# The first part is dedicated to the theoretical computations:
PREC_MU = 20  # precision of the plot.
ALPHA = 2  # fix the parameter alpha.

# We choose the interval of parameter to execute the function.
init_mu = np.linspace(-0.5, 0.5, PREC_MU)

# Storage of the result:
theo_sol = np.ones((PREC_MU, 3))


for i in range(PREC_MU):
    try:
        mu = init_mu[i]
        theo_sol[i, :] = final_function(ALPHA, mu)
    except ValueError:
        theo_sol[i] = 0
        print("probleme")


# The second part is dedicated to the empirical computations:


B_size = 500

emp_sol = np.zeros((PREC_MU, 3))

for i in range(PREC_MU):
    mu = init_mu[i]
    print("Iteration:", i+1, '/', PREC_MU)

    emp_sol[i, :] = empirical_prop(
        B_size=B_size, alpha=ALPHA, mu=mu, mc_prec=200)


# The following is dedicated to the plot.
# For more information, see functions.py
# Remark: theo_sol[:, 0] -> p
# theo_sol[:, 1] -> m
# theo_sol[:, 2] -> sigma


def plot_mu(title):
    if title == 'p':
        fig = plt.figure(1, figsize=(10, 6))
        plt.plot(init_mu, theo_sol[:, 0], 'k', label=r'$p$')
        plt.plot(init_mu, emp_sol[:, 0], 'k*', label='$\widehat{p}$')
        plt.xlabel(r"Interaction drift ($\mu$)", fontsize=15)
        plt.ylabel(r"Proportion of the surviving species ($p)$", fontsize=15)
        plt.ylim(0.8, 1)
        plt.show()
    elif title == 'm':
        fig = plt.figure(1, figsize=(10, 6))
        plt.plot(init_mu, theo_sol[:, 1], 'k', label=r'$m$')
        plt.plot(init_mu, emp_sol[:, 1], 'k*', label='$\widehat{m}$')
        plt.xlabel(r"Interaction drift ($\mu$)", fontsize=15)
        plt.ylabel(r"Mean of the surviving species ($m*)$", fontsize=15)
        plt.show()
    elif title == 'sigma':
        fig = plt.figure(1, figsize=(10, 6))
        plt.plot(init_mu, theo_sol[:, 2], 'k', label=r'$\sigma$')
        plt.plot(init_mu, emp_sol[:, 2], 'k*', label='$\widehat{\sigma}$')
        plt.xlabel(r"Interaction drift ($\mu$)", fontsize=15)
        plt.ylabel(
            r"Root mean square of the surviving species ($\sigma*)$", fontsize=15)
        plt.show()
    else:
        fig = 'Choose p,m or sigma'

    return fig


plot_mu('p')
plot_mu('m')
plot_mu('sigma')
