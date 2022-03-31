# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 09:42:31 2022

@author: Maxime Clenet

This file is dedicated to the evolution of the Hill number of order 1 for a Monte
Carlo experiment on a Lotka-Volterra dynamics.

"""

# Importations of the main packages and functions:

import numpy as np
import matplotlib.pyplot as plt
from functions import lv_dynamics, alpha_abrupt


# %%
# Choice of the initial condition:
n = 100  # Dimension
MC_PREC = 500  # Number of MC experiments
x_init = np.random.random(n)  # Initial condition
NBR_IT = 600  # Number of iterations
TAU = 0.1  # Time step

hill = np.ones(NBR_IT)
hill_neg = np.ones(NBR_IT)
hill_pos = np.ones(NBR_IT)

x_init = np.random.random(n)*2
for i in range(MC_PREC):

    A = np.random.randn(n, n)

    hill += lv_dynamics(A, alpha_abrupt, mu=0, x_init=x_init,
                        nbr_it=NBR_IT, tau=TAU)[1]
    hill_neg += lv_dynamics(A, alpha_abrupt, mu=-0.4, x_init=x_init,
                            nbr_it=NBR_IT, tau=TAU)[1]
    hill_pos += lv_dynamics(A, alpha_abrupt, mu=0.4, x_init=x_init,
                            nbr_it=NBR_IT, tau=TAU)[1]

hill /= MC_PREC
hill_neg /= MC_PREC
hill_pos /= MC_PREC

x = np.linspace(0, NBR_IT*TAU, NBR_IT)  # x-axis

# Part reserved for the display of the dynamics.

fig = plt.figure(1, figsize=(10, 6))
plt.xlabel("Time (t)", fontsize=15)
plt.ylabel(r"Shannon diversity index $e^H$", fontsize=15)
plt.plot(x, hill, color='k', linestyle='--', label=r"$\mu = 0$")
plt.plot(x, hill_neg, color='k', linestyle=':', label=r"$\mu = -0.4$")
plt.plot(x, hill_pos, color='k', label=r"$\mu = 0.4$")
plt.legend(loc='upper right')
plt.show()


# %%

# Choice of the initial condition:
n = 100  # Dimension
MC_PREC = 500  # Number of MC experiments
x_init = np.random.random(n)  # Initial condition
NBR_IT = 600  # Number of iterations
TAU = 0.1  # Time step

hill = np.ones((MC_PREC, NBR_IT))


x_init = np.random.random(n)*2
for i in range(MC_PREC):

    A = np.random.randn(n, n)

    hill[i, :] = lv_dynamics(A, alpha_abrupt, mu=0, x_init=x_init,
                             nbr_it=NBR_IT, tau=TAU)[1]

hill_mean = np.mean(hill, axis=0)
hill_up = np.quantile(hill, 0.95, axis=0)
hill_down = np.quantile(hill, 0.05, axis=0)


x = np.linspace(0, NBR_IT*TAU, NBR_IT)  # x-axis

# Part reserved for the display of the dynamics.

fig = plt.figure(1, figsize=(10, 6))
plt.xlabel("Time (t)", fontsize=15)
plt.ylabel(r"Shannon diversity index $e^H$", fontsize=15)
plt.plot(x, hill_mean, color='k', linestyle='-', label=r"$mean$")
plt.plot(x, hill_up, color='k', linestyle='--', label=r"$q(0.95)$")
plt.plot(x, hill_down, color='k', linestyle=':', label=r"$q(0.05)$")
plt.legend(loc='upper right')
plt.show()
