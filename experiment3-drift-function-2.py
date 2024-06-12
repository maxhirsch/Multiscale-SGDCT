import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.special import i0, i1
from tqdm import tqdm 
import sys
from pathlib import Path

Path("./Paper Figures/").mkdir(parents=True, exist_ok=True)
Path("./Paper Data/").mkdir(parents=True, exist_ok=True)

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

epsilon_list = [0.1, 0.2]
sigma_list = [0.5, 2]

"""
V_0(x) = x^4/4 - x^2/2;
V_1(x, y) = x^2/2 * cos(y) * \one\{x \in [-2, 2]\}
"""

MOVING_AVERAGE = True
X0 = 0.

T = 10**3 # Large T important!
n = 10**6
dt = T / n

def V1(x, y): # p(x, y)
    return x**2/2 * np.cos(y) * (np.abs(x) <= 2)

def K(x):
    if abs(x) > 2:
        return 1
    else:
        return 1/i0(x**2 / (2*sigma))**2

def V_tilde_prime(x, sigma): # grad V, alpha = (1/4, 1/2)
    if abs(x) > 2:
        return x**3 - x
    else:
        return (x**3 - x) / i0(x**2 / (2*sigma))**2 + x / i0(x**2 / (2*sigma))**3 * i1(x**2 / (2*sigma))

@nb.njit
def seed(a):
    np.random.seed(a)

seed(0)

@nb.njit
def multiscale(epsilon, sigma, X0, T, n):

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
            Y[i] = Y[i-1] - (
                    Y[i-1]**3 - Y[i-1] + (
                        Y[i-1] * np.cos(Y[i-1] / epsilon) - Y[i-1]**2 / (2*epsilon) * np.sin(Y[i-1] / epsilon)
                    ) * (np.abs(Y[i-1]) <= 2)
                ) * dt + sd * np.random.normal()
        
        t_ = t
    
    return Y


"""

Varying parameters!

"""
MOVING_AVERAGE = True

for j, epsilon in enumerate(epsilon_list):
    for k, sigma in enumerate(sigma_list):

        ###########################################################################

        Y = multiscale(epsilon, sigma, X0, T, n)

        ###########################################################################

        if not MOVING_AVERAGE:
            delta = 1
            Y_f = np.zeros(Y.shape[0]) # filtered
            Y_f[0] = 0
            for i in tqdm(range(1, Y.shape[0])):
                Y_f[i] = np.exp(-dt/delta) * Y_f[i-1] + np.exp(-dt/delta) * Y[i-1] * dt
            Y_f = Y_f / delta
        else:
            delta = 1
            Y_f = np.zeros(Y.shape[0])
            Y_f[0] = 0
            shift = int(delta/dt)
            for i in range(1, shift):
                Y_f[i] = 1/(i*dt) * np.sum(Y[:i+1]) * dt
            for i in tqdm(range(shift, Y.shape[0])):
                Y_f[i] = 1/delta * np.sum(Y[i-shift:i]) * dt

        ###########################################################################


        r = 3 # approximation order
        A_array = []
        A_old = np.zeros(r+1)
        A_new = np.zeros(r+1)

        for i, t in tqdm(enumerate(np.linspace(0, T, n+1)[:-1])):
            alpha_t = 0.25 / (1 + t/10)
            Yi = Y[i]
            Yi_f = Y_f[i]

            Y_pows = Yi**np.arange(r+1)
            Yf_pows = Yi_f**np.arange(r+1)

            A_new = A_old + alpha_t * (
                -Yf_pows * (Y[i+1] - Yi) - Yf_pows * np.sum(A_old * Y_pows) * dt
            )

            A_old = np.copy(A_new)
            A_array.append(A_new)

        A_array = np.array(A_array)

        x = np.linspace(-4, 4, 500)#np.linspace(np.min(Y), np.max(Y), 500)
        plt.plot(x, [V_tilde_prime(xi, sigma) for xi in x], label="Exact $b$", linewidth=3)
        plt.plot(x, np.sum(A_new[:, None]*np.array([x**i for i in range(r+1)]), axis=0), label="Approximation $\widetilde b$", linewidth=3)
        #plt.title("Drift Function")
        #plt.xlabel("$x$")
        #plt.ylabel("$\widetilde b(x)$")
        plt.legend()
        if MOVING_AVERAGE:
            plt.savefig(f'./Paper Figures/experiment3-ma-42-{j}-{k}.pdf', bbox_inches='tight')
            with open(f"./Paper Data/experiment3-ma-42-{j}-{k}.npy", 'wb') as f:
                np.save(f, A_new)
        else:
            plt.savefig(f'./Paper Figures/experiment3-exponential-42-{j}-{k}.pdf', bbox_inches='tight')
            with open(f"./Paper Data/experiment3-exponential-42-{j}-{k}.npy", 'wb') as f:
                np.save(f, A_new)
        plt.clf()



"""
V_0(x) = x^2/2;
V_1(x, y) = x^2/2 * cos(y) * \one\{x \in [-2, 2]\}
"""

def V_tilde_prime(x, sigma): # grad V, alpha = (1/4, 1/2)
    if abs(x) > 2:
        return x
    else:
        return x / i0(x**2 / (2*sigma))**2 + x / i0(x**2 / (2*sigma))**3 * i1(x**2 / (2*sigma))


@nb.njit
def multiscale2(epsilon, sigma, X0, T, n):

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
            Y[i] = Y[i-1] - (
                    Y[i-1] + (
                        Y[i-1] * np.cos(Y[i-1] / epsilon) - Y[i-1]**2 / (2*epsilon) * np.sin(Y[i-1] / epsilon)
                    ) * (np.abs(Y[i-1]) <= 2)
                ) * dt + sd * np.random.normal()
        
        t_ = t
    
    return Y

"""

Varying parameters!

"""
MOVING_AVERAGE = True

for j, epsilon in enumerate(epsilon_list):
    for k, sigma in enumerate(sigma_list):

        ###########################################################################

        Y = multiscale2(epsilon, sigma, X0, T, n)

        ###########################################################################

        if not MOVING_AVERAGE:
            delta = 1
            Y_f = np.zeros(Y.shape[0]) # filtered
            Y_f[0] = 0
            for i in tqdm(range(1, Y.shape[0])):
                Y_f[i] = np.exp(-dt/delta) * Y_f[i-1] + np.exp(-dt/delta) * Y[i-1] * dt
            Y_f = Y_f / delta
        else:
            delta = 1
            Y_f = np.zeros(Y.shape[0])
            Y_f[0] = 0
            shift = int(delta/dt)
            for i in range(1, shift):
                Y_f[i] = 1/(i*dt) * np.sum(Y[:i+1]) * dt
            for i in tqdm(range(shift, Y.shape[0])):
                Y_f[i] = 1/delta * np.sum(Y[i-shift:i]) * dt

        ###########################################################################


        r = 3 # approximation order
        A_array = []
        A_old = np.zeros(r+1)
        A_new = np.zeros(r+1)

        for i, t in tqdm(enumerate(np.linspace(0, T, n+1)[:-1])):
            alpha_t = 0.25 / (1 + t/10)
            Yi = Y[i]
            Yi_f = Y_f[i]

            Y_pows = Yi**np.arange(r+1)
            Yf_pows = Yi_f**np.arange(r+1)

            A_new = A_old + alpha_t * (
                -Yf_pows * (Y[i+1] - Yi) - Yf_pows * np.sum(A_old * Y_pows) * dt
            )

            A_old = np.copy(A_new)
            A_array.append(A_new)

        A_array = np.array(A_array)

        x = np.linspace(-4, 4, 500)#np.linspace(np.min(Y), np.max(Y), 500)
        plt.plot(x, [V_tilde_prime(xi, sigma) for xi in x], label="Exact $b$", linewidth=3)
        plt.plot(x, np.sum(A_new[:, None]*np.array([x**i for i in range(r+1)]), axis=0), label="Approximation $\widetilde b$", linewidth=3)
        #plt.title("Drift Function")
        #plt.xlabel("$x$")
        #plt.ylabel("$\widetilde b(x)$")
        plt.legend()
        if MOVING_AVERAGE:
            plt.savefig(f'./Paper Figures/experiment3-ma-2-{j}-{k}.pdf', bbox_inches='tight')
            with open(f"./Paper Data/experiment3-ma-2-{j}-{k}.npy", 'wb') as f:
                np.save(f, A_new)
        else:
            plt.savefig(f'./Paper Figures/experiment3-exponential-2-{j}-{k}.pdf', bbox_inches='tight')
            with open(f"./Paper Data/experiment3-exponential-2-{j}-{k}.npy", 'wb') as f:
                np.save(f, A_new)
        plt.clf()