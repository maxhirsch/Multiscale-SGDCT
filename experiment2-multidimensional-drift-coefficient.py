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

MOVING_AVERAGE = True

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
sigma = 0.5

T = 10**5 # Large T important!
n = 10**8
dt = T / n

Y = multiscale(A1, A2, L, epsilon, sigma, X0, T, n)

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

Y_fpow = np.column_stack((-Y_f**3, Y_f))
Y_pow = np.column_stack((-Y**3, Y))
A_array = np.zeros((n+1, 2))

A1_old = 0.
A1_new = 0. 
A2_old = 0.
A2_new = 0. 

A_old = np.zeros(2)
A_new = np.zeros(2)
i = 0
for t in tqdm(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    Yfi = Y_f[i]

    A_new = A_old - alpha_t * Y_fpow[i] * np.dot(A_old, Y_pow[i]) * dt + alpha_t * Y_fpow[i] * (Y[i+1] - Yi)
    A_old[:] = A_new[:]
    A_array[i+1] = A_new

    i += 1

K = L**2 / (integrate.quad(lambda y: np.exp(-np.sin(2 * np.pi / L * y)/sigma), 0, L)[0] * integrate.quad(lambda y: np.exp(np.sin(2 * np.pi / L * y)/sigma), 0, L)[0])

plt.plot(dt * np.arange(A_array.shape[0]), A_array[:, 0], color=blue)
plt.axhline(y=K*A1, color=orange)
plt.xlabel("Time $t$")
plt.ylabel("Estimate $(\widehat{A}_t^\\varepsilon)_1$ of $A_1$")
plt.show()
print(A_array[-5:, 0])

plt.plot(dt * np.arange(A_array.shape[0]), A_array[:, 1], color=blue)
plt.axhline(y=K*A2, color=orange)
plt.xlabel("Time $t$")
plt.ylabel("Estimate $(\widehat{A}_t^\\varepsilon)_2$ of $A_2$")
plt.show()
print(A_array[-5:, 1])


KA = K*np.array([1., 2.])
estimates_dim1 = int(n // 10000)+1#min(n+1, 10000000)
# try this for many trials
@nb.njit(parallel=True)
def another_potential_repeated_estimates(n_trials):
    estimates = np.zeros((estimates_dim1, 2))
    for trial in nb.prange(n_trials):
        Y = multiscale(A1, A2, L, epsilon, sigma, X0, T, n)

        if not MOVING_AVERAGE:
            delta = 1
            Y_f = np.zeros(Y.shape[0]) # filtered
            Y_f[0] = 0
            for i in range(1, Y.shape[0]):
                Y_f[i] = np.exp(-dt/delta) * Y_f[i-1] + np.exp(-dt/delta) * Y[i-1] * dt
            Y_f = Y_f / delta
        else:
            delta = 1
            Y_f = np.zeros(Y.shape[0])
            Y_f[0] = 0
            shift = int(delta/dt)
            for i in range(1, shift):
                Y_f[i] = 1/(i*dt) * np.sum(Y[:i+1]) * dt
            for i in range(shift, Y.shape[0]):
                Y_f[i] = 1/delta * np.sum(Y[i-shift:i]) * dt

        Y_fpow = np.column_stack((-Y_f**3, Y_f))
        Y_pow = np.column_stack((-Y**3, Y))
        A_array = np.zeros((n+1, 2))

        A_old = np.zeros(2)
        A_new = np.zeros(2)
        i = 0
        j = 0
        for t in np.linspace(0, T, n+1)[:-1]:
            if i % 10000 == 0:#if n+1 - i <= estimates_dim1:
                estimates[j] = estimates[j] + (A_old - KA)**2
                j += 1
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            Yfi = Y_f[i]

            A_new = A_old - alpha_t * Y_fpow[i] * np.dot(A_old, Y_pow[i]) * dt + alpha_t * Y_fpow[i] * (Y[i+1] - Yi)
            A_old[:] = A_new[:]
            A_array[i+1] = A_new

            i += 1
        
        estimates[-1] = estimates[-1] + (A_old - KA)**2
    estimates = estimates / n_trials
    estimates = np.sqrt(estimates)
    return estimates

estimates = another_potential_repeated_estimates(50)
"""
plt.hist(estimates[:, -1, 0])
plt.axvline(x=K*A1, color=orange)
plt.show()
#print("Mean estimate (A1) =", np.mean(estimates[:, 0]))

plt.hist(estimates[:, -1, 1])
plt.axvline(x=K*A2, color=orange)
plt.show()
#print("Mean estimate (A2) =", np.mean(estimates[:, 1]))
"""

times = np.linspace(0, T, estimates.shape[0])#(T/n)*np.arange(0, estimates.shape[0], 10000)
#times = T - times[-1] + times
"""
plt.loglog(times, np.sqrt(np.mean((estimates - KA)**2, axis=0))[:, 0], color=blue, label="$L^2$ Error")
plt.loglog(times, times**(-0.5), color=orange, label="$\mathcal{O}(T^{-1/2})$ Reference")
plt.legend()
plt.xlabel("Time $t$")
plt.ylabel("Approximate $L^2$ Error of $(\widehat{A}_t^\\varepsilon)_1$ from $A_1$")
plt.show()

plt.loglog(times, np.sqrt(np.mean((estimates - KA)**2, axis=0))[:, 1], color=blue, label="$L^2$ Error")
plt.loglog(times, times**(-0.5), color=orange, label="$\mathcal{O}(T^{-1/2})$ Reference")
plt.legend()
plt.xlabel("Time $t$")
plt.ylabel("Approximate $L^2$ Error of $(\widehat{A}_t^\\varepsilon)_2$ from $A_2$")
plt.show()
"""

plt.loglog(times, estimates[:, 0], color=blue, label="$L^2$ Error")
plt.loglog(times, times**(-0.5), color=orange, label="$\mathcal{O}(T^{-1/2})$ Reference")
plt.legend()
plt.xlabel("Time $t$")
plt.ylabel("Approximate $L^2$ Error of $(\widehat{A}_t^\\varepsilon)_1$ from $A_1$")
plt.show()

plt.loglog(times, estimates[:, 1], color=blue, label="$L^2$ Error")
plt.loglog(times, times**(-0.5), color=orange, label="$\mathcal{O}(T^{-1/2})$ Reference")
plt.legend()
plt.xlabel("Time $t$")
plt.ylabel("Approximate $L^2$ Error of $(\widehat{A}_t^\\varepsilon)_2$ from $A_2$")
plt.show()