import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm 

plt.rcParams['font.size'] = 12
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
epsilon = 0.1
sigma = 0.1

T = 10**4
n = 10**7
dt = T / n

Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)

########################################################
#
#    Look at effect of filter width on convergence
#
########################################################

#Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)
K = L**2 / (integrate.quad(lambda y: np.exp(-np.sin(2 * np.pi / L * y)/sigma), 0, L)[0] * integrate.quad(lambda y: np.exp(np.sin(2 * np.pi / L * y)/sigma), 0, L)[0])
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
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            Y_fi = Y_f[i]
            A_new = A_old - alpha_t * A_old * Yi*Y_fi * dt - alpha_t * Y_fi * (Y[i+1] - Yi)
            A_old = A_new

        estimates_exp.append(A_new)

    # plot with alpha and A = K*alpha as dotted lines
    plt.plot(delta_exponents, estimates_exp, c=blue, label="Exponential Filter")
    plt.axhline(y=alpha, c=orange, linestyle="--")
    plt.text(0.05, alpha-0.02, "$\\alpha$")
    plt.axhline(y=K*alpha, c=orange, linestyle="--")
    plt.text(0.05, K*alpha+0.01, "$A$")
    plt.xlabel("$\zeta^{-1}$ (where $\\varepsilon = \delta^{\zeta}$)")
    plt.ylabel("Estimate $\widehat{A}^\\varepsilon_T$")
    plt.show()


#########################################################
#
#   Look at single SGDCT then distribution of estimates
#
#########################################################

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
    Y_fi = Y_f[i]
    A_new = A_old - alpha_t * A_old * Yi*Y_fi * dt - alpha_t * Y_fi * (Y[i+1] - Yi)
    A_old = A_new
    A_array.append(A_new)

K = L**2 / (integrate.quad(lambda y: np.exp(-np.sin(2 * np.pi / L * y)/sigma), 0, L)[0] * integrate.quad(lambda y: np.exp(np.sin(2 * np.pi / L * y)/sigma), 0, L)[0])
plt.plot(dt*np.arange(len(A_array)), A_array, color=blue, label="Estimate of $A$")
plt.axhline(y=K*alpha, color=orange)
plt.xlabel("Time $t$")
plt.ylabel("Estimate $\widehat A^\\varepsilon_t$")
plt.show()

print(A_array[-1], "Expected A:", K*alpha)

# try this for many trials
@nb.njit(parallel=True)
def multiscale_repeated_estimates(n_trials):
    estimates = np.zeros((n_trials, n+1))
    for trial in nb.prange(n_trials):
        Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)
        Y_f = np.zeros(Y.shape[0]) # filtered
        Y_f[0] = 0
        for i in range(1, Y.shape[0]):
            Y_f[i] = Y_f[i-1] * np.exp(-dt) + np.exp(-dt) * Y[i-1] * dt

        A_old = 0.
        A_new = 0. 
        estimates[trial, 0] = 0.
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            estimates[trial, i] = A_old
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            Y_fi = Y_f[i]
            A_new = A_old - alpha_t * A_old * Yi*Y_fi * dt - alpha_t * Y_fi * (Y[i+1] - Yi)
            A_old = A_new
        
        estimates[trial, -1] = A_old
        
    return estimates

estimates = multiscale_repeated_estimates(50)
plt.hist(estimates[:, -1])
plt.axvline(x=K*alpha, color=orange)
plt.title("Estimate $\widehat A_T^\\varepsilon$")
plt.show()
print("Mean estimate =", np.mean(estimates[:, -1]), "Standard deviation =", np.std(estimates[:, -1]))

A = K*alpha

plt.loglog(np.linspace(0, T, n+1), np.sqrt(np.mean((estimates - A)**2, axis=0)), color=blue, label="$L^2$ Error")
plt.loglog(np.linspace(0, T, n+1), np.linspace(0, T, n+1)**(-0.5), color=orange, label="$\mathcal{O}(T^{-1/2})$ Reference")
plt.legend()
plt.xlabel("Time $t$")
plt.ylabel("Approximate $L^2$ Error of $\widehat{A}_t^\\varepsilon$ from $A$")
plt.show()