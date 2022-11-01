import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats
from tqdm import tqdm 

@nb.njit
def seed(a):
    np.random.seed(a)

seed(0)

@nb.njit
def ornstein_uhlenbeck(A, X0, T, n):
    """
    Perform Euler-Maruyama for the SDE
        dX_t = -A*X_t dt + dW_t, t \in [0, T]
    with initial condition X_t(0) = X0 for n+1 equi-spaced 
    times 0 = t_0 < t_1 < ... < t_n = T.
    """

    dt = T / n
    sqrt_dt = np.sqrt(dt)
    Y = np.zeros(n+1)
    t_ = -dt

    for i, t in enumerate(np.linspace(0, T, n+1)):
        if i == 0:
            Y[i] = X0
        else:
            Y[i] = Y[i-1] - A * Y[i-1] * dt + sqrt_dt * np.random.normal()
        
        t_ = t
    
    return Y

"""
Ornstein-Uhlenbeck Process
"""

X0 = 0.
A = 5
T = 100000. # Large T important! increase...
n = 1000000
dt = T / n

Y = ornstein_uhlenbeck(A, X0, T, n)
plt.plot(np.linspace(0, T, n+1), Y)
plt.show()

A_array = []

A_old = 100.
A_new = 0. 
for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    A_new = A_old - alpha_t * A_old * Yi**2 * dt - alpha_t * Yi * (Y[i+1] - Yi)
    A_old = A_new
    A_array.append(A_new)

plt.plot(np.log(np.arange(len(A_array))+1), A_array, label="Estimate of $A$")
plt.axhline(y=A, color='r', label="$A$")
plt.show()

print(A_array[-5:])

# try this for many trials
@nb.njit(parallel=True)
def ornstein_uhlenbeck_repeated_estimates(n_trials):
    estimates = np.zeros((n_trials, np.linspace(0, T, n+1)[:-1].shape[0]))
    for trial in nb.prange(n_trials):
        Y = ornstein_uhlenbeck(A, X0, T, n)

        A_old = 100.
        A_new = 0. 
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            A_new = A_old - alpha_t * A_old * Yi**2 * dt - alpha_t * Yi * (Y[i+1] - Yi)
            A_old = A_new
        
            estimates[trial, i] = A_new
    return estimates

estimates = ornstein_uhlenbeck_repeated_estimates(1000)
plt.hist(estimates[:, -1])
plt.axvline(x=A, color='r')
plt.show()
print("Mean estimate =", np.mean(estimates[:, -1]))

Sigma_bar = 10**2 / (2 * (10 - A))
sqrt_Sigma_bar = np.sqrt(Sigma_bar)

plt.hist(np.sqrt(T) * (estimates[:, -1] - A), density=True, bins=30)
plt.plot(np.linspace(-3*sqrt_Sigma_bar, 3*sqrt_Sigma_bar, 100), stats.norm.pdf(np.linspace(-3*sqrt_Sigma_bar, 3*sqrt_Sigma_bar, 100), 0, sqrt_Sigma_bar))
plt.show()

print(Sigma_bar, np.std(np.sqrt(T) * (estimates[:, -1] - A))**2)

array = np.mean(np.abs(estimates - A), axis=0)
plt.loglog(np.arange(array.shape[0])*dt, array)
plt.loglog((1+np.arange(array.shape[0]))*dt, 1/np.sqrt((1+np.arange(array.shape[0]))*dt))
plt.show()