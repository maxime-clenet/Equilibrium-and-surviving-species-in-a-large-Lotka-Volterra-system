# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:41:47 2022

@author: Maxime Clenet
"""


# Importations of the main packages and functions:

import numpy as np
import matplotlib.pyplot as plt
from functions import final_function, alpha_abrupt


# %%
# Representation of the variation of the interaction strength through time.

x = np.linspace(0, 59.9, 100)
N = 10
y_a = np.ones(100)


for i, value in enumerate(x):
    y_a[i] = alpha_abrupt(value)

fig = plt.figure(1, figsize=(10, 6))
plt.plot(x, y_a, color='k')
plt.axhline(np.sqrt(2*np.log(10)),color='black',linestyle='--',label = 'Feasibility threshold')
plt.ylim(0,3.5)
axes = plt.gca()
axes.yaxis.set_ticks([0,0.5,1,1.5,2,np.sqrt(2*np.log(10)),2.5,3,3.5])
axes.yaxis.set_ticklabels(('0','0.5','1','1.5','2',r'$\sqrt{2log(10)}$','2.5','3','3.5'),color = 'black', fontsize = 10)
plt.xlabel("Time (t)", fontsize=15)
plt.ylabel(r"Variation of the interaction strength $\alpha(t)$", fontsize=15)
plt.legend(loc='upper right')
plt.show()


# %%
# Representation of the proportion of surviving species through time.

x = np.linspace(0, 59.9, 100)
N = 10
p_a = np.ones(100)
y_a = np.ones(100)

MU = 0

for i, value in enumerate(x):
    p_a[i] = final_function(alpha_abrupt(value), MU)[0]


fig = plt.figure(1, figsize=(10, 6))
plt.plot(x, p_a, color='k', linestyle='--')
plt.ylim(0,1.1)
plt.xlabel("Time (t)", fontsize=15)
plt.ylabel(r"Variation of the proportion of surviving species $p(t)$", fontsize=15)
plt.show()
