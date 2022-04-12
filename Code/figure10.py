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

# Comparison of the estimator as a function of the strength
# of the interactions (alpha).

# Choice of the initial condition:
N_SIZE = 100  # Dimension
MC_PREC = 50  # Number of MC experiments
MC_ALPHA = 20  # Number of step of the x-axis
MU = 0  # Choice of the parameter MU

# initialisation of the parameter mu
init_alpha = np.linspace(1.5, 3, MC_ALPHA)
# The bounds can be changed, however the result of
# the fixed point can be affected
# (see function.py to change the initial conditions if necessary).

emp_hill = np.zeros(MC_ALPHA)
theo_hill = np.zeros(MC_ALPHA)

for compt, alpha in enumerate(init_alpha):
    # Empirical part:
    s_emp = 0
    for j in range(MC_PREC):
        B = np.random.randn(N_SIZE, N_SIZE)/(np.sqrt(N_SIZE)*alpha)+MU/N_SIZE
        s_emp += theoretical_hill_lcp(B)
    emp_hill[compt] = s_emp/MC_PREC
    # Theoretical part:
    x = final_function(alpha=alpha, mu=MU)
    theo_hill[compt] = N_SIZE*x[0]/2*(3-(x[2]/x[1])**2)

# Part reserved for the display:

fig = plt.figure(1, figsize=(10, 6))
plt.xlabel(r"Interaction strength ($\alpha$)", fontsize=15)
plt.ylabel(r"Hill number of order one ($e^{H'}$)", fontsize=15)
plt.plot(init_alpha, theo_hill, color='k', label=r"Theoretical")
plt.plot(init_alpha, emp_hill, 'k*', label=r"Empirical")
plt.legend(loc='lower right')
plt.show()


# %%

# Comparison of the estimator as a function of the drift
# of the interactions (mu).


# Choice of the initial condition:
N_SIZE = 100  # Dimension
MC_PREC = 100  # Number of MC experiments
MC_MU = 20  # Number of step of the x-axis
ALPHA = 2  # Choice of parameter alpha

init_mu = np.linspace(-0.4, 0.4, MC_MU)  # initialisation of the parameter mu
# The bounds can be changed, however the result of
# the fixed point can be affected
# (see function.py to change the initial conditions if necessary).


emp_hill_mu = np.zeros(MC_MU)
theo_hill_mu = np.zeros(MC_MU)

for compt, mu in enumerate(init_mu):
    # Empirical part:
    s_emp = 0
    for j in range(MC_PREC):
        B = np.random.randn(N_SIZE, N_SIZE)/(np.sqrt(N_SIZE)*ALPHA)+mu/N_SIZE
        s_emp += theoretical_hill_lcp(B)
    emp_hill[compt] = s_emp/MC_PREC
    # Theoretical part:
    x = final_function(alpha=ALPHA, mu=mu)
    theo_hill[compt] = N_SIZE*x[0]/2*(3-(x[2]/x[1])**2)


# Part reserved for the display:

fig = plt.figure(1, figsize=(10, 6))
plt.xlabel(r"Interaction drift ($\mu$)", fontsize=15)
plt.ylabel(r"Hill number of order one ($e^{H'}$)", fontsize=15)
plt.plot(init_alpha, theo_hill, 'k', label=r"Theoretical")
plt.plot(init_alpha, emp_hill, 'k*',  label=r"Empirical")
plt.legend(loc='upper right')
plt.ylim(80, 95)
plt.show()
