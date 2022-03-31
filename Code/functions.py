# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:05:10 2022

@author: Maxime Clenet

Implementation of the function needed to run the figure file.
In particular, the two main functions are:
- f -> distribution of abondances
- final_function -> resolution of the system of 3 equations/3 unknowns.
"""

# Importation of the packages:
import numpy as np
import scipy.stats as stats
from scipy import optimize
from lemkelcp import lemkelcp


def e_cond(p, m, sigma, alpha, mu):
    """
    This function is dependent from the gamma function.

    Conditional mean of the normal distribution.
    See the article for more informations.


    Parameters
    ----------
    p : float
        Proportion of persistent species.
    m : float
        Mean of the persistent species.
    sigma : float
        Root square mean of the persistent species.
    alpha : float
        Parameter of the model - Interaction strength.
    mu : float
        Parameter of the model - Interaction drift.

    Returns
    -------
    float
        Conditional mean associated to the system.

    """

    # The value delta is similar in the article.
    delta = alpha/(sigma*np.sqrt(p))*(1+mu*p*m)

    p_1 = np.exp(-delta**2/2)
    p_2 = 1-stats.norm.cdf(-delta)

    return (1/np.sqrt(2*np.pi))*p_1/p_2


def e2_cond(p, m, sigma, alpha, mu):
    """
    This function is dependent from the gamma function.

    Conditional mean of the square of the normal distribution.
    See the article for more informations.


    Parameters
    ----------
    p : float
        Proportion of persistent species.
    m : float
        Mean of the persistent species.
    sigma : float
        Root square mean of the persistent species.
    alpha : float
        Parameter of the model - Interaction strength.
    mu : float
        Parameter of the model - Interaction drift.

    Returns
    -------
    float
        Conditional mean associated to the system.

    """

    # The value delta is similar in the article.
    delta = alpha/(sigma*np.sqrt(p))*(1+mu*p*m)
    p_1 = np.exp(-delta**2/2)
    p_2 = 1-stats.norm.cdf(-delta)

    return (1/np.sqrt(2*np.pi))*-delta*p_1/p_2+1


# The following equations correspond to the system of equations


def gamma_1(p, m, sigma, alpha, mu):
    """
    Equation (1) associated to the persistent species.

    Parameters
    ----------
    p : float
        Proportion of persistent species.
    m : float
        Mean of the persistent species.
    sigma : float
        Root square mean of the persistent species.
    alpha : float
        Parameter of the model - Interaction strength.
    mu : float
        Parameter of the model - Interaction drift.

    Returns
    -------
    float
        Fixed point equation (1).

    """
    return sigma*np.sqrt(p)*stats.norm.ppf(1-p)+alpha*(1+mu*p*m)


def gamma_2(p, m, sigma, alpha, mu):
    """
    Equation (2) associated to the mean of the abundances.

    Parameters
    ----------
    p : float
        Proportion of persistent species.
    m : float
        Mean of the persistent species.
    sigma : float
        Root square mean of the persistent species.
    alpha : float
        Parameter of the model - Interaction strength.
    mu : float
        Parameter of the model - Interaction drift.

    Returns
    -------
    float
        Fixed point equation (2).

    """
    a_3 = 1+mu*p*m

    b_3 = sigma*np.sqrt(p)/alpha

    return a_3+b_3*e_cond(p, m, sigma, alpha, mu)-m


def gamma_3(p, m, sigma, alpha, mu):
    """
    Equation (3) associated to the square root of the abundances

    Parameters
    ----------
    p : float
        Proportion of persistent species.
    m : float
        Mean of the persistent species.
    sigma : float
        Root square mean of the persistent species.
    alpha : float
        Parameter of the model - Interaction strength.
    mu : float
        Parameter of the model - Interaction drift.


    Returns
    -------
    float
        Fixed point equation (3).

    """

    a_2 = (1+mu*p*m)**2

    b_2 = 2*(1+mu*p*m)*(sigma*np.sqrt(p)/alpha)

    c_2 = sigma**2*p/alpha**2

    return a_2+b_2*e_cond(p, m, sigma, alpha, mu)+c_2*e2_cond(p, m, sigma, alpha, mu)-sigma**2


def sys_gamma(x, alpha, mu):
    """
    Creation of the system of equation using the three gamma functions.

    Parameters
    ----------
    x : list
        x[O] correspond à p
        x[1] correspond à m
        x[2] correspond to sigma
    alpha : float
        Parameter of the model - Interaction strength.
    mu : float
        Parameter of the model - Interaction drift.

    Returns
    -------
    Fixed point equation.

    """

    return (gamma_1(x[0], x[1], x[2], alpha, mu),
            gamma_2(x[0], x[1], x[2], alpha, mu), gamma_3(x[0], x[1], x[2], alpha, mu))


def final_function(alpha, mu):
    """
    Resolution and solution of the system gamma.
    To resolve the system, we use the python package optimize.root.

    Important remark:
    If you have problems with precision when obtaining figures,
    you probably need to manage the settings of the fixed point resolution.


    Parameters
    ----------
    alpha : float
        Parameter of the model - Interaction strength.
    mu : float
        Parameter of the model - Interaction drift.
    Returns
    -------
    p : float
        Proportion of persistent species.
    m : float
        Mean of the persistent species.
    sigma : float
        Root square mean of the persistent species.

    """

    # Management of the fixed point resolution:
    (p, m, sigma) = optimize.root(
        sys_gamma, [0.999, 1.01, 1.01], args=(alpha, mu,)).x

    return p, m, sigma


def density_abundance(x, p, m, sigma, alpha, mu):
    """
    Density function of the persistent species.

    Parameters
    ----------
    v : float
        x-value of the density function.
    p : float
        Proportion of persistent species.
    m : float
        Mean of the persistent species.
    sigma : float
        Root square mean of the persistent species.
    alpha : float
        Parameter of the model - Interaction strength.
    mu : float
        Parameter of the model - Interaction drift.

    Returns
    -------
    float
        y-value of the function.

    """

    const = alpha/(sigma*np.sqrt(p))

    delta = alpha/(sigma*np.sqrt(p))*(1+mu*p*m)

    a = np.exp(-(const*x-delta)**2/2)*const
    b = (1-stats.norm.cdf(-delta))*np.sqrt(2*np.pi)
    return a/b


def zero_LCP(A):
    """
    This function resolve the LCP problem of our model.
    If a solution exist, this function return the properties
    of the solution i.e:
    - proportion of persistent species,
    - variance of the persistent species,
    - mean of the persistent species.

    Parameters
    ----------
    A : numpy.ndarray(n,n),
        Corresponds to the matrix of interactions.

    Returns
    -------
    If this solution exists, the function return: (En pourcentage)
    I/N : Proportion of surviving species.

    m : Mean of the surviving species.

    sigma: Root mean square of the surviving species.

    """
    A_SIZE = A.shape[0]
    q = np.ones(A_SIZE)
    M = -np.eye(A_SIZE)+A
    sol = lemkelcp.lemkelcp(-M, -q, maxIter=10000)

    res_LCP = sol[0]

    res_LCP_pos = res_LCP[res_LCP != 0]
    I = len(res_LCP_pos)

    m = sum(res_LCP_pos)/I
    sigma = np.sqrt(sum(res_LCP_pos**2)/I)
    return (I/A_SIZE, m, sigma)


def empirical_prop(B_size=200, alpha=2, mu=0, mc_prec=100):
    """
    For a large number of matrix (mc_prec) of size (B_size), an empirical
    estimator of the parameter are given using a MC experiment.


    Parameters
    ----------
    B_size : int, optional
        Dimension of the model. The default is 200.
    alpha : float, optional
        Parameter of the model - Interaction strength. The default is 2.
    mu : float, optional
        Parameter of the model - Interaction drift. The default is 0.
    mc_prec : int, optional
        Precision of the MC experiment. The default is 100.

    Returns
    -------
    np.mean(S_p) : float,
        Proportion of surviving species estimator.

    np.mean(S_m) : float,
        Mean of the surviving species estimator.

    np.mean(S_sigma) : float,
        Root mean square of the surviving species estimator.

    """
    S_p = np.zeros(mc_prec)
    S_sigma = np.zeros(mc_prec)
    S_m = np.zeros(mc_prec)

    for i in range(mc_prec):
        const = (mu*alpha/np.sqrt(B_size))
        B = (np.random.randn(B_size, B_size)+const)*(1/(np.sqrt(B_size)*alpha))

        # Verification of the spectral norm condition:
        # while np.sqrt(np.max(np.linalg.eigvals(np.dot(np.transpose(B+np.transpose(B)), B+np.transpose(B))))) >= 2:
        #   B = (np.random.randn(B_size, B_size)+const) * \
        #       (1/(np.sqrt(B_size)*alpha))

        (S_p[i], S_m[i], S_sigma[i]) = zero_LCP(B)

    return np.mean(S_p), np.mean(S_m), np.mean(S_sigma)


def alpha_abrupt(t, intensity=1):
    """
    Correspond to the function alpha(t) of the sudden incident.

    Parameters
    ----------
    t : float,
        Time.
    intensity : float,
        Intensity of the step of the function. The default is 1.

    Returns
    -------
    float
        The function \alpha(t).

    """
    if int(t/30) % 2 != 0:
        return np.sqrt(2)

    return np.sqrt(2)+2*intensity


def f_LV_alpha(x, A, alpha, mu):
    """
    Function used in the RK scheme to approximate the dynamics of the LV EDO.

    Parameters
    ----------
    x : float,
        x_k in the iterative scheme.
    A : numpy.ndarray(n,n),
        Non-normalized matrix of interactions.
    alpha : float,
        Interaction strength.
    mu : float,
        Interaction drift.

    Returns
    -------
    x : float.

    """

    n = A.shape[0]
    x = np.dot(np.diag(x), (np.ones(n)-x +
               np.dot(A*(1/(np.sqrt(n)*alpha))+mu/n, x)))
    return x


def hill(x):
    """

    Parameters
    ----------
    x : numpy.array,
        Vector of abundances.

    Returns
    -------
    float
        Hill number of order 1.

    """
    H = 0
    S = sum(x)
    x_norm = x/S
    for n in x_norm:
        H -= n*np.log(n)

    return np.exp(H)


def lv_dynamics(A, alpha, mu, x_init, nbr_it, tau):
    """
    Runge-Kutta Scheme

    Parameters
    ----------
    A : numpy.ndarray,
        Non-normalized matrix of interactions.
    alpha : function,
        Function of the interaction strength.
    mu : float,
        Interaction drift.
    x_init : numpy.array,
        Initial condition.
    nbr_it : int,
        Number of iterations.
    tau : float,
        Time step.

    Returns
    -------
    sol_dyn : numpy.ndarray,
        Line i corresponds to the values of the dynamics of species i.
    v_hill : numpy.array,
        Vector of the Hill number of order 1 of the system at each time step.

    """
    x = x_init  # x_0 initial condition

    compt = 0  # count of the iterations

    # Storage of the evolution of each species.
    sol_dyn = np.eye(A.shape[0], nbr_it)

    v_hill = np.zeros(nbr_it)  # Hill number for each time step
    # RK scheme: (explicite version)
    while compt < nbr_it:

        t = tau*compt

        f1 = f_LV_alpha(x, A, alpha(t), mu)
        f2 = f_LV_alpha(x+tau*0.5*f1, A, alpha(t), mu)
        f3 = f_LV_alpha(x+tau*0.5*f2, A, alpha(t), mu)
        f4 = f_LV_alpha(x+tau*f3, A, alpha(t), mu)

        x = x+tau*(f1+2*f2+2*f3+f4)/6

        for i in range(A.shape[0]):
            sol_dyn[i, compt] = x[i]
        v_hill[compt] = hill(x)
        compt = compt+1

    return sol_dyn, v_hill


def theoretical_hill_lcp(B):
    """
    Theoretical hill number by calculating the LCP solution of the matrix B

    Parameters
    ----------
    B : numpy.ndarray,
        Matrix of interactions.

    Returns
    -------
    float
        Hill number of order 1.

    """

    q = np.ones(B.shape[0])
    M = -np.eye(B.shape[0])+B
    sol = lemkelcp.lemkelcp(-M, -q, maxIter=10000)

    res_LCP = sol[0]

    res_LCP_pos = res_LCP[res_LCP != 0]

    H = 0
    S = sum(res_LCP_pos)
    res_LCP_pos_norm = res_LCP_pos/S

    for n in res_LCP_pos_norm:
        H -= n*np.log(n)

    return np.exp(H)
