# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

Theoretical approximation versus empirical simulations of the Hill
number of order 1. 
"""


# Importations of the main packages and functions:

import numpy as np
import matplotlib.pyplot as plt
from functions import theoretical_hill_lcp, final_function

# %%
# Choice of the initial condition:
n = 100  # Dimension
MC_PREC = 100  # Number of MC experiments
MC_ALPHA = 20


init_alpha = np.linspace(1.5, 3, MC_ALPHA)

emp_hill = np.zeros(MC_ALPHA)
theo_hill = np.zeros(MC_ALPHA)

compt = 0
for alpha in init_alpha:
    print(alpha)
    S_emp = 0
    for j in range(MC_PREC):
        B = np.random.randn(n, n)/(np.sqrt(n)*alpha)
        S_emp += theoretical_hill_lcp(B)
    emp_hill[compt] = S_emp/MC_PREC
    x = final_function(alpha=alpha, mu=0)

    theo_hill[compt] = n*x[0]/2*(3-(x[2]/x[1])**2)
    compt += 1


fig = plt.figure(1, figsize=(10, 6))
plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
plt.ylabel(r"Hill number of order one ($e^{H'}$)", fontsize=15)
plt.plot(init_alpha, emp_hill, color='k', linestyle='--', label=r"Empirical")
plt.plot(init_alpha, theo_hill, color='k', linestyle=':', label=r"Theoretical")
plt.legend(loc='lower right')
plt.show()


# %%

# Choice of the initial condition:
n = 100  # Dimension
MC_PREC = 100  # Number of MC experiments
MC_MU = 20


init_mu = np.linspace(-0.4, 0.4, MC_MU)

emp_hill_mu = np.zeros(MC_MU)
theo_hill_mu = np.zeros(MC_MU)
alpha = 2
compt = 0
for mu in init_mu:
    S_emp = 0
    for j in range(MC_PREC):
        B = np.random.randn(n, n)/(np.sqrt(n)*alpha)+mu/n
        S_emp += theoretical_hill_lcp(B)
    emp_hill[compt] = S_emp/MC_PREC
    x = final_function(alpha=2, mu=mu)

    theo_hill[compt] = n*x[0]/2*(3-(x[2]/x[1])**2)
    compt += 1


fig = plt.figure(1, figsize=(10, 6))
plt.xlabel(r"Interaction drift ($\mu$)", fontsize=15)
plt.ylabel(r"Hill number of order one ($e^{H'}$)", fontsize=15)
plt.plot(init_alpha, emp_hill, color='k', linestyle='--', label=r"Empirical")
plt.plot(init_alpha, theo_hill, color='k', linestyle=':', label=r"Theoretical")
plt.legend(loc='upper right')
plt.ylim(80, 95)
plt.show()
