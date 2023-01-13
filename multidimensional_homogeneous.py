import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm 
import time

@nb.njit(cache=True)
def seed(a):
    np.random.seed(a)


@nb.njit(cache=True)
def another_potential(A1, A2, X0, T, n):
    """
    Perform Euler-Maruyama for the SDE
        dX_t = -A*(X_t^3-X_t) dt + dW_t, t \in [0, T]
    with initial condition X_t(0) = X0 for n+1 equi-spaced 
    times 0 = t_0 < t_1 < ... < t_n = T.
    """

    dt = T / n
    sqrt_dt = np.sqrt(dt)
    Y = np.zeros(n+1)
    Y[0] = X0

    for i in range(1, n+1):
        Y[i] = Y[i-1] - (A1 * Y[i-1]**3 - A2 * Y[i-1]) * dt + sqrt_dt * np.random.normal()
            
    return Y


@nb.njit(parallel=True, cache=True)
def another_potential_repeated_estimates(n_trials):
    estimates = np.zeros((n_trials, 2))
    for trial in nb.prange(n_trials):
        Y = another_potential(A1, A2, X0, T, n)

        A_old = np.zeros(2)
        A_new = np.zeros(2)
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            A_new = A_old - alpha_t * np.array([-Yi**3, Yi]) * (-A_old[0]*Yi**3 + A_old[1]*Yi) * dt + alpha_t * np.array([-Yi**3, Yi]) * (Y[i+1] - Yi)
            A_old[:] = A_new[:]
        
        estimates[trial, 0] = A_new[0]
        estimates[trial, 1] = A_new[1]
    return estimates


@nb.njit(parallel=True, cache=True)
def another_potential_repeated_estimates_MLE(n_trials):
    estimates = np.zeros((n_trials, 2))
    for trial in nb.prange(n_trials):
        Y = another_potential(A1, A2, X0, T, n)
        dVY = np.stack((Y**3, -Y))
        M = (1/T) * dVY @ dVY.T * dt
        h = (1/T) * np.sum(dVY[:, :-1] * (Y[1:] - Y[:-1]), axis=1)
        MLE = -np.linalg.solve(M, h)
        
        estimates[trial, 0] = MLE[0]
        estimates[trial, 1] = MLE[1]
    return estimates


if __name__=="__main__":
    seed(0)

    X0 = 0.
    A1 = 1
    A2 = 2

    T = 10**5 # Large T important!
    n = 10**6
    dt = T / n

    Y = another_potential(A1, A2, X0, T, n)

    plt.plot(np.linspace(0, T, n+1), Y)
    plt.show()

    A1_array = []
    A2_array = []

    A1_old = 0.
    A1_new = 0. 
    A2_old = 0.
    A2_new = 0. 
    for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
        A1_array.append(A1_new)
        A2_array.append(A2_new)

        alpha_t = 0.1 / (1 + t/10)
        Yi = Y[i]
        A1_new = A1_old - alpha_t * (-Yi**3) * (-A1_old*Yi**3 + A2_old*Yi) * dt + alpha_t * (-Yi**3) * (Y[i+1] - Yi)
        A2_new = A2_old - alpha_t * (Yi) * (-A1_old*Yi**3 + A2_old*Yi) * dt + alpha_t * (Yi) * (Y[i+1] - Yi)

        A1_old = A1_new
        A2_old = A2_new

    plt.semilogx(dt * np.arange(len(A1_array)), A1_array)
    plt.axhline(y=A1, c='r')
    plt.xlabel("Time")
    plt.ylabel("Estimate of $A_1$")
    plt.show()

    plt.semilogx(dt * np.arange(len(A2_array)), A2_array)
    plt.axhline(y=A2, c='r')
    plt.xlabel("Time")
    plt.ylabel("Estimate of $A_2$")
    plt.show()

    estimates = another_potential_repeated_estimates(200)

    plt.hist(estimates[:, 0])
    plt.axvline(x=A1, c='r')
    plt.xlabel("Estimate of $A_1$")
    plt.show()

    plt.hist(estimates[:, 1])
    plt.axvline(x=A2, c='r')
    plt.xlabel("Estimate of $A_2$")
    plt.show()


    trials = 10
    times_sgdct = np.zeros(trials)
    times_mle = np.zeros(trials)
    estimates_sgdct = np.zeros((2, trials))
    estimates_mle = np.zeros((2, trials))

    for trial in tqdm(range(trials)):
        Y = another_potential(A1, A2, X0, T, n)
        
        # SGDCT Computation

        start = time.time()
        
        A1_old = 0.
        A1_new = 0. 
        A2_old = 0.
        A2_new = 0. 
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 0.1 / (1 + t/10)
            Yi = Y[i]
            A1_new = A1_old - alpha_t * (-Yi**3) * (-A1_old*Yi**3 + A2_old*Yi) * dt + alpha_t * (-Yi**3) * (Y[i+1] - Yi)
            A2_new = A2_old - alpha_t * (Yi) * (-A1_old*Yi**3 + A2_old*Yi) * dt + alpha_t * (Yi) * (Y[i+1] - Yi)

            A1_old = A1_new
            A2_old = A2_new
        
        end = time.time()
        times_sgdct[trial] = end - start
        estimates_sgdct[0, trial] = A1_new 
        estimates_sgdct[1, trial] = A2_new

        # MLE Computation

        start = time.time()
        
        dVY = np.stack((Y**3, -Y))
        M = (1/T) * dVY @ dVY.T * dt
        h = (1/T) * np.sum(dVY[:, :-1] * (Y[1:] - Y[:-1]), axis=1)
        MLE = -np.linalg.solve(M, h)
        
        end = time.time()
        times_mle[trial] = end - start
        estimates_mle[0, trial] = MLE[0] 
        estimates_mle[1, trial] = MLE[1]

    
    print(f"SGDCT Average Time (Over {trials} trials)", np.mean(times_sgdct))
    plt.hist(estimates_sgdct[0])
    plt.axvline(x=A1, color='r')
    plt.show()
    print("Mean estimate (A1) =", np.mean(estimates_sgdct[0]))
    print("SD Estimate (A1) =", np.std(estimates_sgdct[0]))

    plt.hist(estimates_sgdct[1])
    plt.axvline(x=A2, color='r')
    plt.show()
    print("Mean estimate (A2) =", np.mean(estimates_sgdct[1]))
    print("SD Estimate (A2) =", np.std(estimates_sgdct[1]))

    
    print("MLE Average Time (Over 10 trials):", np.mean(times_mle))
    plt.hist(estimates_mle[0])
    plt.axvline(x=A1, color='r')
    plt.show()
    print("Mean estimate (A1) =", np.mean(estimates_mle[0]))
    print("SD Estimate (A1) =", np.std(estimates_mle[0]))

    plt.hist(estimates_mle[1])
    plt.axvline(x=A2, color='r')
    plt.show()
    print("Mean estimate (A2) =", np.mean(estimates_mle[1]))
    print("SD Estimate (A2) =", np.std(estimates_mle[1]))