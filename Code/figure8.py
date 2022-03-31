# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet


This file is dedicated to the simulations in the case of a alpha
depending of the time.

"""

# Importations of the main packages and functions:

import numpy as np
import matplotlib.pyplot as plt
from functions import lv_dynamics, alpha_abrupt


# %%

# Choice of the initial condition:
n = 10  # Dimension
A = np.random.randn(n, n)  # Matrix of interaction
x_init = np.random.random(n)  # Initial condition
NBR_IT = 60000  # Number of iterations
TAU = 0.001  # Time step
MU = 0  # Interaction drift

sol_dyn = lv_dynamics(A, alpha_abrupt, mu=MU, x_init=x_init,  # y-axis
                      nbr_it=NBR_IT, tau=TAU)[0]

x = np.linspace(0, NBR_IT*TAU, NBR_IT)  # x-axis

# Part reserved for the display of the dynamics.

fig = plt.figure(1, figsize=(10, 6))
for i in range(sol_dyn.shape[0]):
    if sol_dyn[i, -1] > sol_dyn[i, 10000]:
        plt.plot(x, sol_dyn[i, :], color='k', linestyle='--')
    elif sol_dyn[i, -1] < 10**(-2):
        plt.plot(x, sol_dyn[i, :], color='k', linestyle=':')
    else:
        plt.plot(x, sol_dyn[i, :], color='k')


plt.xlabel("Time (t)", fontsize=15)
plt.ylabel("Abundance ($x_i$)", fontsize=15)
#plt.legend(loc='upper right')
plt.show()
