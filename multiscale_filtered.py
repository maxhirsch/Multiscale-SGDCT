import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm 
import sys

MOVING_AVERAGE = True
delta_experiment = False

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
            Y[i] = Y[i-1] + (-alpha * Y[i-1] - (1/epsilon) * (2*np.pi/L) * np.cos(2*np.pi/L * Y[i-1]/epsilon)) * dt + sd * np.random.normal()
    
    return Y

X0 = 0
alpha = 1#3.14
L = 2 * np.pi
epsilon = 0.1
sigma = 1.0

T = 6*10**4
n = 6*10**7
dt = T / n

Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)
plt.plot(np.linspace(0, T, n+1), Y)
plt.show()

########################################################
#
#    Look at effect of filter width on convergence
#
########################################################

#Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)
K = L**2 / (integrate.quad(lambda y: np.exp(-np.sin(2 * np.pi / L * y)/sigma), 0, L)[0] * integrate.quad(lambda y: np.exp(np.sin(2 * np.pi / L * y)/sigma), 0, L)[0])
if delta_experiment:
    delta_exponents = np.linspace(0, 3, 13)
    estimates_exp = []
    estimates_avg = []

    for p in tqdm(delta_exponents):
        delta = epsilon**p

        # Exponential filtered path
        Y_f = np.zeros(Y.shape[0]) # filtered
        Y_f[0] = 0
        for i in range(1, Y.shape[0]):
            Y_f[i] = np.exp(-dt/delta) * Y_f[i-1] + np.exp(-dt/delta) * Y[i-1] * dt
        Y_f = Y_f / delta

        A_old = 0.
        A_new = 0. 
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            Y_fi = Y_f[i]
            A_new = A_old - alpha_t * A_old * Yi*Y_fi * dt - alpha_t * Y_fi * (Y[i+1] - Yi)
            A_old = A_new

        estimates_exp.append(A_new)

        # Moving average path
        Y_f = np.zeros(Y.shape[0])
        Y_f[0] = 0
        shift = int(delta/dt)
        for i in range(1, shift):
            Y_f[i] = 1/(i*dt) * np.sum(Y[:i+1]) * dt
        for i in range(shift, Y.shape[0]):
            Y_f[i] = 1/delta * np.sum(Y[i-shift:i]) * dt
        
        
        A_old = 0.
        A_new = 0. 
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            Y_fi = Y_f[i]
            A_new = A_old - alpha_t * A_old * Yi*Y_fi * dt - alpha_t * Y_fi * (Y[i+1] - Yi)
            A_old = A_new

        estimates_avg.append(A_new)
        print(delta, estimates_exp[-1], estimates_avg[-1])

    # plot with alpha and A = K*alpha as dotted lines
    plt.plot(delta_exponents, estimates_exp, label="Exponential Filter")
    plt.plot(delta_exponents, estimates_avg, label="Moving Average")
    plt.axhline(y=alpha, linestyle="--")
    plt.text(0.05, alpha-0.02, "$\\alpha$")
    plt.axhline(y=K*alpha, linestyle="--")
    plt.text(0.05, K*alpha+0.01, "$A$")
    plt.xlabel("$p$ (Exponent in $\delta = \\varepsilon^{p}$)")
    plt.ylabel("SGDCT Estimate of $A$")
    plt.legend()
    plt.show()


#########################################################
#
#   Look at single SGDCT then distribution of estimates
#
#########################################################

if not MOVING_AVERAGE:
    delta = 1
    Y_f = np.zeros(Y.shape[0]) # filtered
    Y_f[0] = 0
    for i in tqdm(range(1, Y.shape[0])):
        Y_f[i] = np.exp(-dt/delta) * Y_f[i-1] + np.exp(-dt/delta) * Y[i-1] * dt
        #Y_f[i] = Y_f[i-1] * np.exp(-dt) + np.exp(-dt) * Y[i-1] * dt
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


plt.plot(np.linspace(0, T, n+1), Y)
plt.plot(np.linspace(0, T, n+1), Y_f)
plt.show()



A_array = []
A_old = 0.
A_new = 0. 
for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 1 / (1 + t/10)
    Yi = Y[i]
    Y_fi = Y_f[i]
    A_new = A_old - alpha_t * A_old * Yi*Y_fi * dt - alpha_t * Y_fi * (Y[i+1] - Yi)
    A_old = A_new
    A_array.append(A_new)

K = L**2 / (integrate.quad(lambda y: np.exp(-np.sin(2 * np.pi / L * y)/sigma), 0, L)[0] * integrate.quad(lambda y: np.exp(np.sin(2 * np.pi / L * y)/sigma), 0, L)[0])
plt.semilogx(dt*np.arange(len(A_array)), A_array, label="Estimate of $A$")
plt.axhline(y=K*alpha, color='r')
plt.xlabel("Time")
plt.ylabel("Estimate of $A$")
plt.show()

sys.exit(0)

print(A_array[-1], "Expected A:", K*alpha)


# try this for many trials
@nb.njit(parallel=True)
def multiscale_repeated_estimates(n_trials):
    estimates = np.zeros(n_trials)
    for trial in nb.prange(n_trials):
        Y = multiscale(alpha, L, epsilon, sigma, X0, T, n)
        Y_f = np.zeros(Y.shape[0]) # filtered
        Y_f[0] = 0
        for i in range(1, Y.shape[0]):
            Y_f[i] = Y_f[i-1] * np.exp(-dt) + np.exp(-dt) * Y[i-1] * dt

        A_old = 100.
        A_new = 0. 
        for i, t in enumerate(np.linspace(0, T, n+1)[:-1]):
            alpha_t = 1 / (1 + t/10)
            Yi = Y[i]
            Y_fi = Y_f[i]
            A_new = A_old - alpha_t * A_old * Yi*Y_fi * dt - alpha_t * Y_fi * (Y[i+1] - Yi)
            A_old = A_new
        
        estimates[trial] = A_new
    return estimates

estimates = multiscale_repeated_estimates(200)
plt.hist(estimates)
plt.axvline(x=K*alpha, color='r')
plt.show()
print("Mean estimate =", np.mean(estimates))



