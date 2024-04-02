import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm 
import sys
import winsound

plt.rcParams['font.size'] = 12

MOVING_AVERAGE = True
X0 = 0.
epsilon = 0.05#0.1
sigma = 2#1.5#0.5

T = 10**3 # Large T important!
n = 8*10**6
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
print(Y[:5])

#plt.plot(np.linspace(0, T, n+1), Y)
#plt.show()

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


#plt.plot(np.linspace(0, T, n+1), Y)
#plt.plot(np.linspace(0, T, n+1), Y_f)
#plt.show()

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
"""
plt.semilogx(dt * np.arange(len(A_array)), A_array[:, 0], label="Estimate of $A1$")
#plt.axhline(y=A1, color='r', label="$A1$")
plt.xlabel("Time")
plt.ylabel("Estimate of $A_1$")
plt.show()
print(A_array[-5:, 0])

plt.semilogx(dt * np.arange(len(A_array)), A_array[:, 1], label="Estimate of $A2$")
#plt.axhline(y=A2, color='r', label="$A2$")
plt.xlabel("Time")
plt.ylabel("Estimate of $A_2$")
plt.show()
print(A_array[-5:, 1])
"""

duration = 1500  # milliseconds
freq = 660  # Hz
winsound.Beep(freq, duration)

x = np.linspace(np.min(Y), np.max(Y), 500)
plt.plot(x, [V_tilde_prime(xi) for xi in x], label="Exact $b$")
plt.plot(x, np.sum(A_new[:, None]*np.array([x**i for i in range(r+1)]), axis=0), label="Approximation $\widetilde b$")
plt.title("Drift Function")
plt.xlabel("$x$")
plt.ylabel("$\widetilde b(x)$")
plt.legend()
plt.show()












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
