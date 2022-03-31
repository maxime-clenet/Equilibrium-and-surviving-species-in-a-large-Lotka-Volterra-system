# -*- coding: utf-8 -*-
"""
@author: Maxime Clenet

Representation of the phase diagram i.e the set of admissible parameter A. 
"""

# Importation of the packages

import numpy as np
import matplotlib.pyplot as plt


# %%
# The phase diagram can be computed following the definition of the set of admissible parameter A

# Initialisation of the alpha bound:
limit_alpha = np.linspace(np.sqrt(2), 4, 1000)
limit_mu = np.array([])

limit_perturbed = np.array([])

for alpha in limit_alpha:
    delta = (1-2/alpha**2)
    limit_mu = np.append(limit_mu, 1/2+np.sqrt(delta)/2)
    limit_perturbed = np.append(limit_perturbed, 1/(np.sqrt(2)*alpha))

# Display of the figure:
fig = plt.figure(1, figsize=(10, 6))
ax = fig.add_subplot(111)
# Corresponds to the left vertical bound:
plt.plot([np.sqrt(2), np.sqrt(2)], [-0.5, 0.5], linestyle='--', color='k')
# Corresponds to the upper bound:
plt.plot(limit_alpha, limit_mu, linestyle='--', color='k')

plt.plot(limit_alpha, limit_perturbed, color='k')

# Added a hatched area to represent the admissible area:
plt.fill_between(limit_alpha, limit_mu, color='#0000003d')
plt.xlabel(r"$a$", fontsize=15)
plt.ylabel(r"$m$", fontsize=15)
plt.ylim(0, 1)
plt.xlim(1, 4)

plt.text(0.65, 0.55, r'Zone $\mathcal{B}$', horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes, fontsize=20, color='k')

plt.text(0.4, 0.15, r'Zone $\mathcal{C}$', horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes, fontsize=20, color='k')


# plt.title(
#    r'Evolution of the mean condition ($\mu_{max}$) in function of $\alpha$')
plt.show()
plt.close()
