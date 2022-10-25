# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

This file is dedicated to study the optimal stability 
threshold of the equilibrium point.

"""

# Importation of the packages:

import numpy as np
import matplotlib.pyplot as plt
import math
from functions import run


# %%

n = 500
prec = 200
S = 0
init_alpha = np.linspace(0.6, 1.4, 20)

res = np.zeros(20)
nan = np.zeros(20)
for i, alpha in enumerate(init_alpha):
    print(i)
    S = 0
    c = 0
    compt_nan = 0
    for j in range(prec):
        x_init = np.random.random(n)
        traj = run(n, x_init, alpha)
        if traj == 'burst':
            compt_nan += 1
        else:

            #        a = np.mean(np.std(traj[1:, -50:], axis=0) /
            #                    np.mean(traj[1:, -50:], axis=0))
            a = np.mean(np.std(traj[1:, -50:], axis=0))
            if math.isnan(a) == False:
                S += a
                c += 1

    if c != 0:
        res[i] = S/c
    else:
        res[i] = 0
    nan[i] = compt_nan


fig = plt.figure(1, figsize=(10, 6))
plt.plot(init_alpha[2:], res[2:], color='black')
plt.xlabel(r"Interaction strength $\alpha$", fontsize=15)
plt.ylabel("Standard deviation (SD)", fontsize=15)
# plt.yscale('log')
#plt.title('Feasible equilibrium (n: 100, meanA: 0, std: 3)')
#plt.legend(loc='upper right')
plt.axvline(1/np.sqrt(2), color='black',
            linestyle='dotted', label='Threshold')
plt.axvline(1, color='black',
            linestyle='dashed', label='Threshold')
plt.show()
plt.close()

fig = plt.figure(1, figsize=(10, 6))
plt.plot(init_alpha, nan/200, 'black')
plt.xlabel(r"Interaction strength $\alpha$", fontsize=15)
plt.ylabel("Proportion of unbounded growth", fontsize=15)
plt.axvline(1/np.sqrt(2), color='black',
            linestyle='dotted', label='Threshold')
plt.axvline(1, color='black',
            linestyle='dashed', label='Threshold')
plt.show()
plt.close()
