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
def multiscale(A1, A2, L, epsilon, sigma, X0, T, n):
    """
    Perform Euler-Maruyama for the SDE
        dX_t = -A*(X_t^3-X_t) dt + dW_t, t \in [0, T]
    with initial condition X_t(0) = X0 for n+1 equi-spaced 
    times 0 = t_0 < t_1 < ... < t_n = T.
    """

    dt = T / n
    sqrt_dt = np.sqrt(dt)
    sqrt_2sigma = np.sqrt(2 * sigma)
    sd = sqrt_dt * sqrt_2sigma
    Y = np.zeros(n+1)
    t_ = -dt

    for i, t in enumerate(np.linspace(0, T, n+1)):
        if i == 0:
            Y[i] = X0
        else:
            Y[i] = Y[i-1] - (A1 * Y[i-1]**3 - A2 * Y[i-1] + (1 / epsilon) * (2 * np.pi / L) * np.cos(2 * np.pi / L * Y[i-1] / epsilon)) * dt + sd * np.random.normal()
        
        t_ = t
    
    return Y

X0 = 0.
A1 = 1
A2 = 2
L = 2 * np.pi
epsilon = 0.1
sigma = 1.0

T = 10**5 # Large T important!
n = 10**8
dt = T / n

Y = multiscale(A1, A2, L, epsilon, sigma, X0, T, n)

plt.plot(np.linspace(0, T, n+1), Y)
plt.show()

A_array = []

A1_old = 10.
A1_new = 0. 
A2_old = 10.
A2_new = 0. 
for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    A1_new = A1_old - alpha_t * (-Yi**3) * (-A1_old*Yi**3 + A2_old*Yi - (1 / epsilon) * (2 * np.pi / L) * np.cos(2 * np.pi / L * Y[i-1] / epsilon)) * dt + alpha_t * (-Yi**3) * (Y[i+1] - Yi)
    A2_new = A2_old - alpha_t * (Yi) * (-A1_old*Yi**3 + A2_old*Yi - (1 / epsilon) * (2 * np.pi / L) * np.cos(2 * np.pi / L * Y[i-1] / epsilon)) * dt + alpha_t * (Yi) * (Y[i+1] - Yi)

    A1_old = A1_new
    A2_old = A2_new
    A_array.append([A1_new, A2_new])

A_array = np.array(A_array)

plt.plot(np.log(np.arange(len(A_array))+1), A_array[:, 0], label="Estimate of $A1$")
plt.axhline(y=A1, color='r', label="$A1$")
plt.show()
print(A_array[-5:, 0])

plt.plot(np.log(np.arange(len(A_array))+1), A_array[:, 1], label="Estimate of $A2$")
plt.axhline(y=A2, color='r', label="$A2$")
plt.show()
print(A_array[-5:, 1])

if False:
    # try this for many trials
    @nb.njit(parallel=True)
    def another_potential_repeated_estimates(n_trials):
        estimates = np.zeros((n_trials, 2))
        for trial in nb.prange(n_trials):
            Y = another_potential(A1, A2, X0, T, n)

            A1_old = 10.
            A1_new = 0. 
            A2_old = 10.
            A2_new = 0. 
            for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
                alpha_t = 1 / (1 + t/10)
                Yi = Y[i]
                A1_new = A1_old - alpha_t * (-Yi**3) * (-A1_old*Yi**3 + A2_old*Yi) * dt + alpha_t * (-Yi**3) * (Y[i+1] - Yi)
                A2_new = A2_old - alpha_t * (Yi) * (-A1_old*Yi**3 + A2_old*Yi) * dt + alpha_t * (Yi) * (Y[i+1] - Yi)

                A1_old = A1_new
                A2_old = A2_new
            
            print(A1_new, A2_new)
            estimates[trial, 0] = A1_new
            estimates[trial, 1] = A2_new
        return estimates

    estimates = another_potential_repeated_estimates(200)
    plt.hist(estimates[:, 0])
    plt.axvline(x=A1, color='r')
    plt.show()
    print("Mean estimate (A1) =", np.mean(estimates[:, 0]))

    plt.hist(estimates[:, 1])
    plt.axvline(x=A2, color='r')
    plt.show()
    print("Mean estimate (A2) =", np.mean(estimates[:, 1]))
