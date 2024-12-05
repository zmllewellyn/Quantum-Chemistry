import numpy as np
import scipy as sp
import numba
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

# Atom 1 paramaters
alpha1 = -0.6
beta1 = -0.1

# Atom 2 parameters
alpha2 = -0.6
beta2 = -0.

# Length of chain (divisible by 2)
L = 10

''' Matrix Equation '''
# F Matrix
F = np.zeros((L, L))
# S Matrix
S = np.identity(L)

for i in range(L):
    for j in range(L):
        if i == j and i%2 == 0:
            F[i,j] = alpha1
        elif i == j and i%2 == 1:
            F[i,j] = alpha2
        elif abs(i-j) == 1 and i%2 == 0:
            F[i,j] = beta1
        elif abs(i-j) == 1 and i%2 == 1:
            F[i,j] = beta2

sol = np.linalg.solve(S, F)
eval, evec = np.linalg.eigh(sol)

''' Plot Energy Levels '''
fig, ax = plt.subplots()

ax.set_ylabel("Energy Level ($E_{h}$)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

evals_degen = Counter(np.round(eval, decimals=6))
max_degen = 0
for ev in list(evals_degen.keys()):
    ax.barh(ev, height=0.005, width=evals_degen[ev], color='r')

    if evals_degen[ev] > max_degen:
        max_degen = evals_degen[ev]

ax.set_xlabel("Degeneracy")
ax.set_xticks(np.arange(1, max_degen + 1))
plt.show()

''' Chain 3D Density '''
fid = 40
cutoff = 0.05

xrange = np.linspace(0 - 2, L + 2, fid)
yrange = np.linspace(-2, 2, fid)
zrange = np.linspace(-2, 2, fid)

a = 1.
post = []
for i in range(L):
    post.append((i * a, 0., 0.))


@numba.njit
def benz_3D(fid, cutoff, xrange, yrange, zrange, post, alpha1, beta1, alpha2, beta2, evec):
    def lam(i, alpha, beta):
        return alpha * (beta ** (i - 1))

    def basis_func(i, alpha, beta, pos, atom_pos):
        r = np.sqrt((pos[0] - atom_pos[0]) ** 2 + (pos[1] - atom_pos[1]) ** 2 + (pos[2] - atom_pos[2]) ** 2)
        # theta = np.arccos((pos[2]-atom_pos[2])/r)
        theta = np.arctan2(np.sqrt((pos[0] - atom_pos[0]) ** 2 + (pos[1] - atom_pos[1]) ** 2), pos[2])
        phi = np.arctan2((pos[1] - atom_pos[1]), (pos[0] - atom_pos[0]))
        # li = abs(alpha*(beta**(i-1)))
        li = 2.
        return 0.5 * np.sqrt(3 / np.pi) * np.sqrt(((2 * li) ** 5) / (4 * 3 * 2 * 1)) * r * np.exp(-li * r) * np.cos(
            theta)

    dist = np.zeros((fid, fid, fid))
    size_val = np.zeros((fid, fid, fid))

    for i in range(fid):
        for j in range(fid):
            for k in range(fid):
                x = xrange[i]
                y = yrange[j]
                z = zrange[k]

                for m in range(len(post)):
                    for n in range(len(post)):
                        f_m = basis_func(0, alpha1, beta1, [x, y, z], post[m])
                        f_n = basis_func(0, alpha1, beta1, [x, y, z], post[n])
                        c_m = evec.T[5, m]
                        c_n = evec.T[5, n]
                        dist[i, j, k] += c_m * c_n * f_m * f_n

                if dist[i, j, k] <= cutoff:
                    size_val[i, j, k] = 0
                else:
                    size_val[i, j, k] = 20

    point = []
    for i in range(fid):
        for j in range(fid):
            for k in range(fid):
                x = xrange[i]
                y = yrange[j]
                z = zrange[k]

                point.append([x, y, z])

    dist = np.round(dist, 5)
    point = np.asarray(point)
    return dist, point, size_val


dist, point, size_val = benz_3D(fid, cutoff, xrange, yrange, zrange, post, alpha1, beta1, alpha2, beta2, evec)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

co = dist*(1/np.max(dist))
ax.scatter(point[:,0], point[:,1], point[:,2], c=co, s=size_val, alpha=0.2, cmap='viridis')

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")
ax.set_zlabel("z Position ($a_{0}$)")

ax.azim = 120
ax.elev = 10
plt.show()

''' Chain Electron Density Cross-section '''
fid = 200
z = 0.5
xrange = np.linspace(0 - 2, L + 2, fid)
yrange = np.linspace(-2, 2, fid)

a = 1.
post = []
for i in range(L):
    post.append((i * a, 0., 0.))


@numba.njit
def chain_cross(fid, xrange, yrange, alpha_c, beta_c, post, evec):
    def lam(i, alpha, beta):
        return alpha * (beta ** (i - 1))

    def basis_func(i, alpha, beta, pos, atom_pos):
        r = np.sqrt((pos[0] - atom_pos[0]) ** 2 + (pos[1] - atom_pos[1]) ** 2 + (pos[2] - atom_pos[2]) ** 2)
        # theta = np.arccos((pos[2]-atom_pos[2])/r)
        theta = np.arctan2(np.sqrt((pos[0] - atom_pos[0]) ** 2 + (pos[1] - atom_pos[1]) ** 2), pos[2])
        phi = np.arctan2((pos[1] - atom_pos[1]), (pos[0] - atom_pos[0]))
        # li = abs(alpha*(beta**(i-1)))
        li = 1.627
        return 0.5 * np.sqrt(3 / np.pi) * np.sqrt(((2 * li) ** 5) / (4 * 3 * 2 * 1)) * r * np.exp(-li * r) * np.cos(
            theta)

    dist = np.zeros((fid, fid))
    for i in range(fid):
        for j in range(fid):
            x = xrange[i]
            y = yrange[j]

            for m in range(len(post)):
                for n in range(len(post)):
                    f_m = basis_func(0, alpha_c, beta_c, [x, y, z], post[m])
                    f_n = basis_func(0, alpha_c, beta_c, [x, y, z], post[n])
                    c_m = evec.T[5, m]
                    c_n = evec.T[5, n]
                    dist[i, j] += c_m * c_n * f_m * f_n

    return np.round(dist, 5).T


dist = chain_cross(fid, xrange, yrange, alpha1, beta1, post, evec)

fig, ax = plt.subplots()

vmin = np.min(dist)
vmax = np.max(dist)
cmap = cm.viridis
norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

ax.pcolormesh(xrange, yrange, dist)

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")

ax.set_title("n=5, z=0.5 Cross-section")

fig.colorbar(sm, ax=ax, label="Electron Density $|\\Psi|^{2}$")
plt.show()