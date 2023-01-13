import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm 

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
alpha = 1#3.14
L = 2 * np.pi
epsilon = 0.1
sigma = 0.5

T = 10**4
n = 10**7
dt = T / n

Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)
plt.plot(np.linspace(0, T, n+1), Y)
plt.show()


A_array = []

A_old = 0.
A_new = 0. 
for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    A_new = A_old - alpha_t * A_old * Yi**2 * dt - alpha_t * Yi * (Y[i+1] - Yi)
    A_old = A_new
    A_array.append(A_new)

K = L**2 / (integrate.quad(lambda y: np.exp(-np.sin(2 * np.pi / L * y)/sigma), 0, L)[0] * integrate.quad(lambda y: np.exp(np.sin(2 * np.pi / L * y)/sigma), 0, L)[0])

plt.semilogx(dt * np.arange(len(A_array)), A_array, label="Estimate of $A$")
plt.axhline(y=alpha, color='r', linestyle='--')
plt.axhline(y=K*alpha, color='r', linestyle='-')
plt.text(10**(-3), K*alpha + 0.07, "$A$")
plt.text(10**(-3), alpha - 0.15, "$\\alpha$")
plt.xlabel("Time")
plt.ylabel("Estimate of $A$")
plt.show()

plt.hist(A_array[-100:])
plt.show()



print(A_array[-1], "Expected A:", K*alpha)

# try this for many trials
@nb.njit(parallel=True)
def multiscale_repeated_estimates(n_trials):
    estimates = np.zeros(n_trials)
    for trial in nb.prange(n_trials):
        Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)

        A_old = 0.
        A_new = 0. 
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            A_new = A_old - alpha_t * A_old * Yi**2 * dt - alpha_t * Yi * (Y[i+1] - Yi)
            A_old = A_new
        
        estimates[trial] = A_new
    return estimates

estimates = multiscale_repeated_estimates(200)
plt.hist(estimates)
plt.axvline(x=alpha, c='r', linestyle='--')
#plt.axvline(x=K*alpha, c='r', linestyle='-')
plt.show()
print("Mean estimate =", np.mean(estimates))