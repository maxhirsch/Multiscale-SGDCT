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
def another_potential(A, X0, T, n):
    """
    Perform Euler-Maruyama for the SDE
        dX_t = -A*(X_t^3-X_t) dt + dW_t, t \in [0, T]
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
            Y[i] = Y[i-1] - A * (Y[i-1]**3 - Y[i-1]) * dt + sqrt_dt * np.random.normal()
        
        t_ = t
    
    return Y

X0 = 0.
A = 3.14

T = 10000. # Large T important!
n = 100000
dt = T / n

Y = another_potential(A, X0, T, n)

plt.plot(np.linspace(0, T, n+1), Y)
plt.show()

A_array = []

A_old = 100.
A_new = 0. 
for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    A_new = A_old - alpha_t * A_old * (Yi**3 - Yi)**2 * dt - alpha_t * (Yi**3 - Yi) * (Y[i+1] - Yi)
    A_old = A_new
    A_array.append(A_new)

plt.plot(np.log(np.arange(len(A_array))+1), A_array, label="Estimate of $A$")
plt.axhline(y=A, color='r', label="$A$")
plt.show()
print(A_array[-5:])

# try this for many trials
@nb.njit(parallel=True)
def another_potential_repeated_estimates(n_trials):
    estimates = np.zeros(n_trials)
    for trial in nb.prange(n_trials):
        Y = another_potential(A, X0, T, n)

        A_old = 100.
        A_new = 0. 
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            A_new = A_old - alpha_t * A_old * (Yi**3 - Yi)**2 * dt - alpha_t * (Yi**3 - Yi) * (Y[i+1] - Yi)
            A_old = A_new
        
        estimates[trial] = A_new
    return estimates

estimates = another_potential_repeated_estimates(200)
plt.hist(estimates)
plt.axvline(x=A, color='r')
plt.show()
print("Mean estimate =", np.mean(estimates))
