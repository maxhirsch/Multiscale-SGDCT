import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import colors as mcolors
from matplotlib.collections import StarPolygonCollection
from scipy.special import i0
from tqdm import tqdm 
from pathlib import Path

Path("./Paper Figures/").mkdir(parents=True, exist_ok=True)
Path("./Paper Data/").mkdir(parents=True, exist_ok=True)

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
blue = colors[0]
orange = colors[1]

MOVING_AVERAGE = False#True

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

K = 1/i0(1/sigma)**2

plt.plot(dt * np.arange(A_array.shape[0]), A_array[:, 0], color=blue, linewidth=3)
plt.axhline(y=K*A1, color=orange, linewidth=3)
plt.xlabel("Time $t$")
plt.ylabel("Estimate $(\widehat{A}_t^\\varepsilon)_1$ of $A_1$")
if MOVING_AVERAGE:
    plt.savefig('./Paper Figures/experiment2-A1-sample-ma.pdf', bbox_inches='tight')
    with open("./Paper Data/experiment2-A1-sample-ma.npy", 'wb') as f:
        np.save(f, A_array[:, 0])
else:
    plt.savefig('./Paper Figures/experiment2-A1-sample-exponential.pdf', bbox_inches='tight')
    with open("./Paper Data/experiment2-A1-sample-exponential.npy", 'wb') as f:
        np.save(f, A_array[:, 0])
plt.clf()

plt.plot(dt * np.arange(A_array.shape[0]), A_array[:, 1], color=blue, linewidth=3)
plt.axhline(y=K*A2, color=orange, linewidth=3)
plt.xlabel("Time $t$")
plt.ylabel("Estimate $(\widehat{A}_t^\\varepsilon)_2$ of $A_2$")
if MOVING_AVERAGE:
    plt.savefig('./Paper Figures/experiment2-A2-sample-ma.pdf', bbox_inches='tight')
    with open("./Paper Data/experiment2-A2-sample-ma.npy", 'wb') as f:
        np.save(f, A_array[:, 1])
else:
    plt.savefig('./Paper Figures/experiment2-A2-sample-exponential.pdf', bbox_inches='tight')
    with open("./Paper Data/experiment2-A2-sample-exponential.npy", 'wb') as f:
        np.save(f, A_array[:, 1])
plt.clf()

A_array_ = A_array[1::10000]

points = np.array([A_array_[:, 0], A_array_[:, 1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
times_array = dt * np.arange(A_array_.shape[0]) * 10000 + dt

axs = plt.gca()

# Create a continuous norm to map from data points to colors
norm = mcolors.LogNorm(vmin=times_array.min(), vmax=times_array.max())
lc = LineCollection(segments, cmap='viridis', norm=norm, zorder=1)
# Set the values used for colormapping
lc.set_array(times_array)
lc.set_linewidth(2)
line = axs.add_collection(lc)
plt.colorbar(line, label="Time $t$")

plt.xlim(A_array_[:, 0].min(), A_array_[:, 0].max())
plt.ylim(A_array_[:, 1].min(), A_array_[:, 1].max())
plt.scatter([K*A1], [K*A2], c='r', s=200, marker="*", zorder=10)
plt.xlabel("$(\widehat{A}_t^\\varepsilon)_1$")
plt.ylabel("$(\widehat{A}_t^\\varepsilon)_2$")
if MOVING_AVERAGE:
    plt.savefig('./Paper Figures/experiment2-2D-sample-ma.pdf', bbox_inches='tight')
    with open("./Paper Data/experiment2-2D-sample-ma.npy", 'wb') as f:
        np.save(f, A_array)
else:
    plt.savefig('./Paper Figures/experiment2-2D-sample-exponential.pdf', bbox_inches='tight')
    with open("./Paper Data/experiment2-2D-sample-exponential.npy", 'wb') as f:
        np.save(f, A_array)
plt.clf()