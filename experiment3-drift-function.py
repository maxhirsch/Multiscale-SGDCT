import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm 
import sys
from pathlib import Path

Path("./Paper Figures/").mkdir(parents=True, exist_ok=True)
Path("./Paper Data/").mkdir(parents=True, exist_ok=True)

plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = (6.4, 4.8)

MOVING_AVERAGE = True
X0 = 0.
epsilon = 0.05#0.1
sigma = 2#1.5#0.5

T = 10**2#3 # Large T important!
n = 8*10**5#6
dt = T / n

def V1(x, y): # p(x, y)
    return x**2/2 * np.cos(y) * (np.abs(x) <= 2)

def K(x):
    f1 = lambda y: np.exp(-1/sigma * V1(x, y))
    f2 = lambda y: np.exp(1/sigma * V1(x, y))
    factor1 = integrate.quadrature(f1, 0, 2*np.pi)[0]
    factor2 = integrate.quadrature(f2, 0, 2*np.pi)[0]
    return 4*np.pi**2 / (factor1 * factor2)

def V_tilde_prime(x): # grad V, alpha = (1/4, 1/2)
    return K(x)*(x**3 - x)

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
    print(Y, Y_f)

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

duration = 1500  # milliseconds
freq = 660  # Hz

x = np.linspace(np.min(Y), np.max(Y), 500)
plt.plot(x, [V_tilde_prime(xi) for xi in x], label="Exact $b$", linewidth=3)
plt.plot(x, np.sum(A_new[:, None]*np.array([x**i for i in range(r+1)]), axis=0), label="Approximation $\widetilde b$", linewidth=3)
#plt.title("Drift Function")
plt.xlabel("$x$")
plt.ylabel("$\widetilde b(x)$")
plt.legend()
if MOVING_AVERAGE:
    plt.savefig('./Paper Figures/experiment3-ma.png', bbox_inches='tight')
    plt.savefig('./Paper Figures/experiment3-ma.pdf', bbox_inches='tight')
    plt.savefig('./Paper Figures/experiment3-ma.svg', bbox_inches='tight')
    with open("./Paper Data/experiment3-ma.npy", 'wb') as f:
        np.save(f, A_new)
else:
    plt.savefig('./Paper Figures/experiment3-exponential.png', bbox_inches='tight')
    plt.savefig('./Paper Figures/experiment3-exponential.pdf', bbox_inches='tight')
    plt.savefig('./Paper Figures/experiment3-exponential.svg', bbox_inches='tight')
    with open("./Paper Data/experiment3-exponential.npy", 'wb') as f:
        np.save(f, A_new)
plt.clf()

