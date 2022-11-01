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
epsilon = 0.05#0.1
sigma = 1.0

T = 10000.
n = 10000000
dt = T / n

Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)
plt.plot(np.linspace(0, T, n+1), Y)
plt.show()

Y_f = np.zeros(Y.shape[0]) # filtered
Y_f[0] = 0
for i in range(1, Y.shape[0]):
    Y_f[i] = Y_f[i-1] * np.exp(-dt) + np.exp(-dt) * Y[i-1] * dt

plt.plot(np.linspace(0, T, n+1), Y)
plt.plot(np.linspace(0, T, n+1), Y_f)
plt.show()


A_array = []

A_old = 100.
A_new = 0. 
for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    Y_fi = Y_f[i]
    A_new = A_old - alpha_t * A_old * Yi**2 * dt - alpha_t * Yi * (Y[i+1] - Yi)
    A_old = A_new
    A_array.append(A_new)

plt.plot(np.log(np.arange(len(A_array))+1), A_array, label="Estimate of $A$")
plt.show()

plt.hist(A_array[-100:])
plt.show()

K = L**2 / (integrate.quad(lambda y: np.exp(-np.sin(2 * np.pi / L * y)/sigma), 0, L)[0] * integrate.quad(lambda y: np.exp(np.sin(2 * np.pi / L * y)/sigma), 0, L)[0])

print(A_array[-1], "Expected A:", K*alpha)

# try this for many trials
@nb.njit(parallel=True)
def multiscale_repeated_estimates(n_trials):
    estimates = np.zeros(n_trials)
    for trial in nb.prange(n_trials):
        Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)

        A_old = 100.
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
plt.show()
print("Mean estimate =", np.mean(estimates))