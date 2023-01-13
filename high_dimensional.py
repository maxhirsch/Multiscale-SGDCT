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
def another_potential(A1, A2, A3, A4, A5, A6, X0, T, n):
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
            Y[i] = Y[i-1] - (
                A1 * Y[i-1]**5 + A2 * Y[i-1]**4 + A3 * Y[i-1]**3  + A4 * Y[i-1]**2 + A5 * Y[i-1] + A6
            ) * dt + sqrt_dt * np.random.normal()
        
        t_ = t
    
    return Y

X0 = 0.
A1 = 1
A2 = -1
A3 = -5.25
A4 = 4.75
A5 = 5
A6 = -3

T = 10**3 # Large T important!
n = 10**6
dt = T / n

Y = another_potential(A1, A2, A3, A4, A5, A6, X0, T, n)

plt.plot(np.linspace(0, T, n+1), Y)
plt.show()

A_array = []


A_old = np.zeros(6)
A_new = np.zeros(6)
exponents = 5 - np.arange(6)


A1_old = 0.
A1_new = 0. 
A2_old = 0.
A2_new = 0. 
A3_old = 0.
A3_new = 0. 
A4_old = 0.
A4_new = 0. 
A5_old = 0.
A5_new = 0. 
A6_old = 0.
A6_new = 0. 

i = 0
for t in tqdm(np.linspace(0, T, n+1)[:-1]):
    alpha_t = 0.1 / (1 + t/10)
    Yi = Y[i]

    potential_ = A_old * Yi**exponents
    #A_new = A_old - alpha_t * Yi**exponents * potential_ * dt - alpha_t * Yi**exponents * (Y[i+1] - Yi)
    #A_old = A_new
    #A_array.append(A_new)

    
    potential_ = A1_old*Yi**5 + A2_old*Yi**4 + A3_old*Yi**3 + A4_old*Yi**2 + A5_old*Yi + A6_old
    A1_new = A1_old - alpha_t * (Yi**5) * potential_ * dt - alpha_t * (Yi**5) * (Y[i+1] - Yi)
    A2_new = A2_old - alpha_t * (Yi**4) * potential_ * dt - alpha_t * (Yi**4) * (Y[i+1] - Yi)
    A3_new = A3_old - alpha_t * (Yi**3) * potential_ * dt - alpha_t * (Yi**3) * (Y[i+1] - Yi)
    A4_new = A4_old - alpha_t * (Yi**2) * potential_ * dt - alpha_t * (Yi**2) * (Y[i+1] - Yi)
    A5_new = A5_old - alpha_t * Yi * potential_ * dt - alpha_t * Yi * (Y[i+1] - Yi)
    A6_new = A6_old - alpha_t * potential_ * dt - alpha_t * (Y[i+1] - Yi)

    A1_old = A1_new
    A2_old = A2_new
    A3_old = A3_new
    A4_old = A4_new 
    A5_old = A5_new 
    A6_old = A6_new 
    

    A_array.append([A1_new, A2_new, A3_new, A4_new, A5_new, A6_new])
    i += 1

A_array = np.array(A_array)

y = np.linspace(-5, 5, 1000)
plt.plot(y, - (A1 * y**5 + A2 * y**4 + A3 * y**3  + A4 * y**2 + A5 * y + A6), label="Actual")
plt.plot(y, - (A1_new * y**5 + A2_new * y**4 + A3_new * y**3  + A4_new * y**2 + A5_new * y + A6_new), label="Estimate")
plt.title("Potential")
plt.legend()
plt.show()


plt.semilogx(dt * np.arange(len(A_array)), A_array[:, 0], label="Estimate of $A1$")
plt.axhline(y=A1, color='r', label="$A1$")
plt.show()
print(A_array[-5:, 0])

plt.semilogx(dt * np.arange(len(A_array)), A_array[:, 1], label="Estimate of $A2$")
plt.axhline(y=A2, color='r', label="$A2$")
plt.show()
print(A_array[-5:, 1])
