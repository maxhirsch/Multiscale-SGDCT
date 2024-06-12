import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.special import i0
from tqdm import tqdm 
from pathlib import Path

Path("./Paper Figures/").mkdir(parents=True, exist_ok=True)
Path("./Paper Data/").mkdir(parents=True, exist_ok=True)

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
blue = colors[0]
orange = colors[1]

delta_experiment = True

@nb.njit
def seed(a):
    np.random.seed(a)

seed(0)

@nb.njit
def multiscale(alpha, L, epsilon, sigma, X0, T, n):
    """
    Perform Euler-Maruyama for the multiscale SDE
    with initial condition X_t(0) = X0 for n+1 equi-spaced 
    times 0 = t_0 < t_1 < ... < t_n = T.
    """

    dt = T / n
    sqrt_dt = np.sqrt(dt)
    sqrt_2sigma = np.sqrt(2 * sigma)
    sd = sqrt_dt * sqrt_2sigma
    Y = np.zeros(n+1)

    for i, t in enumerate(np.linspace(0, T, n+1)):
        if i == 0:
            Y[i] = X0
        else:
            Y[i] = Y[i-1] + (-alpha * Y[i-1] - (1/epsilon) * (2*np.pi/L) * np.cos(2*np.pi/L * Y[i-1]/epsilon)) * dt + sd * np.random.normal()
    
    return Y

X0 = 0
alpha = 1
L = 2 * np.pi
epsilon = 0.025
sigma = 0.5

T = 10**3
n = 64*10**7
dt = T / n

Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)

########################################################
#
#    Look at effect of filter width on convergence
#
########################################################

K = 1/i0(1/sigma)**2
if delta_experiment:
    delta_exponents = np.linspace(0.1, 3, 13)
    estimates_exp = []

    for p in tqdm(delta_exponents):
        delta = epsilon**p

        # Exponential filtered path
        Y_f = np.zeros(Y.shape[0]) # filtered
        Y_f[0] = 0
        for i in range(1, Y.shape[0]):
            Y_f[i] = np.exp(-dt/delta) * Y_f[i-1] + np.exp(-dt/delta) * Y[i-1] * dt
        Y_f = Y_f / delta

        A_old = 0.
        A_new = 0. 
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 1 / (1 + t)#1 / (1 + t/10)
            Yi = Y[i]
            Y_fi = Y_f[i]
            A_new = A_old - alpha_t * A_old * Yi*Y_fi * dt - alpha_t * Y_fi * (Y[i+1] - Yi)
            A_old = A_new

        estimates_exp.append(A_new)

    # plot with alpha and A = K*alpha as dotted lines
    plt.plot(delta_exponents, estimates_exp, c=blue, label="Exponential Filter", linewidth=3)
    plt.axhline(y=alpha, c=orange, linestyle="--", linewidth=3)
    plt.text(0.05, alpha-0.05, "$\\alpha$", fontsize=16)
    plt.axhline(y=K*alpha, c=orange, linestyle="--", linewidth=3)
    plt.text(0.05, K*alpha+0.02, "$A$", fontsize=16)
    plt.xlabel("$\zeta^{-1}$ (where $\\varepsilon = \delta^{\zeta}$)")
    plt.ylabel("Estimate $\widehat{A}^\\varepsilon_T$")
    plt.savefig('./Paper Figures/experiment1-theoretical-results-delta.pdf', bbox_inches='tight')
    plt.clf()
    with open("./Paper Data/experiment1-theoretical-results-delta-exponents.npy", 'wb') as f:
        np.save(f, delta_exponents)
    with open("./Paper Data/experiment1-theoretical-results-delta-estimates.npy", 'wb') as f:
        np.save(f, estimates_exp)
