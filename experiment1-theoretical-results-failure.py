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
            Y[i] = Y[i-1] + (-alpha * Y[i-1] - (1 / epsilon) * (2 * np.pi / L) * np.cos(2 * np.pi / L * Y[i-1] / epsilon)) * dt + sd * np.random.normal()
    
    return Y

X0 = 0
alpha = 1
L = 2 * np.pi
epsilon = 0.1
sigma = 0.5

T = 5*10**4
n = 5*10**7
dt = T / n

Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)

A_array = []

A_old = 0.
A_new = 0. 
for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    A_new = A_old - alpha_t * A_old * Yi**2 * dt - alpha_t * Yi * (Y[i+1] - Yi)
    A_old = A_new
    A_array.append(A_new)

K = 1/i0(1/sigma)**2

plt.plot(dt * np.arange(len(A_array)), A_array, c=blue, label="Estimate of $A$", linewidth=3)
plt.axhline(y=alpha, color=orange, linestyle='--', linewidth=3)
plt.axhline(y=K*alpha, color=orange, linestyle='-', linewidth=3)
plt.text(0.95*T, K*alpha + 0.07, "$A$", fontsize=16)
plt.text(0.95*T, alpha - 0.19, "$\\alpha$", fontsize=16)
plt.xlabel("Time $t$")
plt.ylabel("Estimate $\widetilde A^\\varepsilon_t$")
plt.savefig('./Paper Figures/experiment1-theoretical-results-failure-sample.pdf', bbox_inches='tight')
plt.clf()
with open("./Paper Data/experiment1-theoretical-results-failure-sample.npy", 'wb') as f:
    np.save(f, A_array)

print("$\widetilde A_T^\\varepsilon$", A_array[-1], "Expected A:", K*alpha)



# now filter
delta = 1
Y_f = np.zeros(Y.shape[0]) # filtered
Y_f[0] = 0
for i in tqdm(range(1, Y.shape[0])):
    Y_f[i] = np.exp(-dt/delta) * Y_f[i-1] + np.exp(-dt/delta) * Y[i-1] * dt
Y_f = Y_f / delta

A_array = []

A_old = 0.
A_new = 0. 
for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    Yfi = Y_f[i]
    A_new = A_old - alpha_t * A_old * Yi*Yfi * dt - alpha_t * Yfi * (Y[i+1] - Yi)
    A_old = A_new
    A_array.append(A_new)

plt.plot(dt * np.arange(len(A_array)), A_array, c=blue, label="Estimate of $A$", linewidth=3)
plt.axhline(y=alpha, color=orange, linestyle='--', linewidth=3)
plt.axhline(y=K*alpha, color=orange, linestyle='-', linewidth=3)
plt.text(0.95*T, K*alpha + 0.07, "$A$", fontsize=16)
plt.text(0.95*T, alpha - 0.1, "$\\alpha$", fontsize=16)
plt.xlabel("Time $t$")
plt.ylabel("Estimate $\widetilde A^\\varepsilon_t$")
plt.savefig('./Paper Figures/experiment1-theoretical-results-failure-sample-filter.pdf', bbox_inches='tight')
plt.clf()
with open("./Paper Data/experiment1-theoretical-results-failure-sample-filter.npy", 'wb') as f:
    np.save(f, A_array)
