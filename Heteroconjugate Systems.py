import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from collections import Counter
import numba

# 2pz orbital energy of carbon
alpha_c = -0.6
# Nearest-neighbour interaction energy for C-C
beta_c = -0.1

''' Benzene Ring '''
# F Matrix
F = np.zeros((6, 6))
# S Matrix
S = np.identity(6)

for i in range(6):
    for j in range(6):
        if i == j:
            F[i,j] = alpha_c
        elif abs(i-j)%4 == 1:
            F[i,j] = beta_c

''' Pyridine '''
# Coefficients
alpha_p1 = -0.5
beta_p1 = -0.8

# F Matrix
F1 = np.zeros((6, 6))
# S Matrix
S1 = np.identity(6)

for i in range(6):
    for j in range(6):
        if i == 0 and j == 0:
            F1[i,j] = alpha_p1
        elif i == 0 and j == 1 or i == 1 and j == 0 or i == 0 and j == 5 or i == 5 and j == 0:
            F1[i,j] = beta_p1
        elif i == j:
            F1[i,j] = alpha_c
        elif abs(i-j)%4 == 1:
            F1[i,j] = beta_c

''' Pyrrole '''
# Coefficients
alpha_p2 = -1.5
beta_p2 = -1.

# F Matrix
F2 = np.zeros((6, 6))
# S Matrix
S2 = np.identity(6)

for i in range(6):
    for j in range(6):
        if i == 0 and j == 0:
            F2[i,j] = alpha_p2
        elif i == 0 and j == 1 or i == 1 and j == 0 or i == 0 and j == 5 or i == 5 and j == 0:
            F2[i,j] = beta_p2
        elif i == j:
            F2[i,j] = alpha_c
        elif abs(i-j)%4 == 1:
            F2[i,j] = beta_c

sol = np.linalg.solve(S, F)
sol1 = np.linalg.solve(S1, F1)
sol2 = np.linalg.solve(S2, F2)

eval, evec = np.linalg.eigh(sol)
eval1, evec1 = np.linalg.eigh(sol1)
eval2, evec2 = np.linalg.eigh(sol2)

''' Plot Energy Levels '''
fig, ax = plt.subplots(1, 3)
fig.tight_layout()

eigen = [eval, eval1, eval2]
eigvec = [evec, evec1, evec2]
heights = [0.001, 0.006, 0.007]

ax[0].set_ylabel("Energy Level ($E_{h}$)")
for i in range(3):
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

    evals_degen = Counter(np.round(eigen[i], decimals=6))
    max_degen = 0
    for ev in list(evals_degen.keys()):
        ax[i].barh(ev, height=heights[i], width=evals_degen[ev], color='r')

        if evals_degen[ev] > max_degen:
            max_degen = evals_degen[ev]

    ax[i].set_xlabel("Degeneracy")
    ax[i].set_xticks(np.arange(1, max_degen + 1))
plt.show()

''' Benzene Electron Density '''
fid = 200
z = 0.2
xrange = np.linspace(-2, 2, fid)
yrange = np.linspace(-2, 2, fid)

post = [(1., 0., 0.), (1 / 2, np.sqrt(3) / 2, 0.), (-1 / 2, np.sqrt(3) / 2, 0.), (-1., 0., 0.),
        (-1 / 2, -np.sqrt(3) / 2, 0.), (1 / 2, -np.sqrt(3) / 2, 0.)]  # Benzene


@numba.njit
def benz_cross(fid, xrange, yrange, alpha_c, beta_c, post, evec):
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
                    c_m = evec.T[0, m]
                    c_n = evec.T[0, n]
                    dist[i, j] += c_m * c_n * f_m * f_n

    return np.round(dist, 5).T


dist = benz_cross(fid, xrange, yrange, alpha_c, beta_c, post, evec)

fig, ax = plt.subplots()

vmin = np.min(dist)
vmax = np.max(dist)
cmap = cm.viridis
norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

ax.pcolormesh(xrange, yrange, dist)

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")

ax.set_title("n=0, z=0.2 Cross-section")

fig.colorbar(sm, ax=ax, label="Electron Density $|\\Psi|^{2}$")

plt.show()

''' Pyridine Electron Density '''


def lam(i, alpha, beta):
    return alpha * (beta ** (i - 1))


def basis_func(i, alpha, beta, pos, atom_pos):
    r = np.sqrt((pos[0] - atom_pos[0]) ** 2 + (pos[1] - atom_pos[1]) ** 2 + (pos[2] - atom_pos[2]) ** 2)
    # theta = np.arccos((pos[2]-atom_pos[2])/r)
    theta = np.arctan2(np.sqrt((pos[0] - atom_pos[0]) ** 2 + (pos[1] - atom_pos[1]) ** 2), pos[2])
    phi = np.arctan2((pos[1] - atom_pos[1]), (pos[0] - atom_pos[0]))
    # li = abs(alpha*(beta**(i-1)))
    li = 1.627
    return 0.5 * np.sqrt(3 / np.pi) * np.sqrt(((2 * li) ** 5) / (4 * 3 * 2 * 1)) * r * np.exp(-li * r) * np.cos(theta)


fid = 200
z = 0.2
xrange = np.linspace(-2, 2, fid)
yrange = np.linspace(-2, 2, fid)

post = [(1., 0., 0.), (1 / 2, np.sqrt(3) / 2, 0.), (-1 / 2, np.sqrt(3) / 2, 0.), (-1., 0., 0.),
        (-1 / 2, -np.sqrt(3) / 2, 0.), (1 / 2, -np.sqrt(3) / 2, 0.)]  # Benzene


@numba.njit
def p1_cross(fid, xrange, yrange, alpha_c, beta_c, alpha_p1, beta_p1, post, evec1):
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

    dist1 = np.zeros((fid, fid))
    for i in range(fid):
        for j in range(fid):
            x = xrange[i]
            y = yrange[j]

            for m in range(len(post)):
                for n in range(len(post)):
                    if m == 0:
                        f_m = basis_func(0, alpha_p1, beta_p1, [x, y, z], post[m])
                    else:
                        f_m = basis_func(0, alpha_c, beta_c, [x, y, z], post[m])

                    if n == 0:
                        f_n = basis_func(0, alpha_p1, beta_p1, [x, y, z], post[n])
                    else:
                        f_n = basis_func(0, alpha_c, beta_c, [x, y, z], post[n])

                    c_m = evec1.T[1, m]
                    c_n = evec1.T[1, n]
                    dist1[i, j] += c_m * c_n * f_m * f_n

    return np.round(dist1, 5).T


dist1 = p1_cross(fid, xrange, yrange, alpha_c, beta_c, alpha_p1, beta_p1, post, evec1)

fig, ax = plt.subplots()

vmin = np.min(dist1)
vmax = np.max(dist1)
cmap = cm.viridis
norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

ax.pcolormesh(xrange, yrange, dist1)

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")

ax.set_title("n=1, z=0.5 Cross-section")

fig.colorbar(sm, ax=ax, label="Electron Density $|\\Psi|^{2}$")

plt.show()

''' Pyrrole Electron Density '''


def lam(i, alpha, beta):
    return alpha * (beta ** (i - 1))


def basis_func(i, alpha, beta, pos, atom_pos):
    r = np.sqrt((pos[0] - atom_pos[0]) ** 2 + (pos[1] - atom_pos[1]) ** 2 + (pos[2] - atom_pos[2]) ** 2)
    # theta = np.arccos((pos[2]-atom_pos[2])/r)
    theta = np.arctan2(np.sqrt((pos[0] - atom_pos[0]) ** 2 + (pos[1] - atom_pos[1]) ** 2), pos[2])
    phi = np.arctan2((pos[1] - atom_pos[1]), (pos[0] - atom_pos[0]))
    # li = abs(alpha*(beta**(i-1)))
    li = 1.627
    return 0.5 * np.sqrt(3 / np.pi) * np.sqrt(((2 * li) ** 5) / (4 * 3 * 2 * 1)) * r * np.exp(-li * r) * np.cos(theta)


fid = 200
z = 0.2
xrange = np.linspace(-2, 2, fid)
yrange = np.linspace(-2, 2, fid)

post = [(1., 0., 0.), (1 / 2, np.sqrt(3) / 2, 0.), (-1 / 2, np.sqrt(3) / 2, 0.), (-1., 0., 0.),
        (-1 / 2, -np.sqrt(3) / 2, 0.), (1 / 2, -np.sqrt(3) / 2, 0.)]  # Benzene


@numba.njit
def p1_cross(fid, xrange, yrange, alpha_c, beta_c, alpha_p2, beta_p2, post, evec2):
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

    dist2 = np.zeros((fid, fid))
    for i in range(fid):
        for j in range(fid):
            x = xrange[i]
            y = yrange[j]

            for m in range(len(post)):
                for n in range(len(post)):
                    if m == 0:
                        f_m = basis_func(0, alpha_p2, beta_p2, [x, y, z], post[m])
                    else:
                        f_m = basis_func(0, alpha_c, beta_c, [x, y, z], post[m])

                    if n == 0:
                        f_n = basis_func(0, alpha_p2, beta_p2, [x, y, z], post[n])
                    else:
                        f_n = basis_func(0, alpha_c, beta_c, [x, y, z], post[n])

                    c_m = evec2.T[0, m]
                    c_n = evec2.T[0, n]
                    dist2[i, j] += c_m * c_n * f_m * f_n

    return np.round(dist2, 5).T


dist2 = p1_cross(fid, xrange, yrange, alpha_c, beta_c, alpha_p2, beta_p2, post, evec2)

fig, ax = plt.subplots()

vmin = np.min(dist2)
vmax = np.max(dist2)
cmap = cm.viridis
norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

ax.pcolormesh(xrange, yrange, dist2)

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")

ax.set_title("n=2, z=0.5 Cross-section")

fig.colorbar(sm, ax=ax, label="Electron Density $|\\Psi|^{2}$")

plt.show()

''' Pyridine 3D Density '''
fid = 60
cutoff = 0.06

xrange = np.linspace(-2, 2, fid)
yrange = np.linspace(-2, 2, fid)
zrange = np.linspace(-2, 2, fid)

post = [(1., 0., 0.), (1 / 2, np.sqrt(3) / 2, 0.), (-1 / 2, np.sqrt(3) / 2, 0.), (-1., 0., 0.),
        (-1 / 2, -np.sqrt(3) / 2, 0.), (1 / 2, -np.sqrt(3) / 2, 0.)]  # Benzene


@numba.njit
def p1_3D(fid, cutoff, xrange, yrange, zrange, post, alpha_c, beta_c, alpha_p1, beta_p1, evec1):
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
                        if m == 0:
                            f_m = basis_func(0, alpha_p1, beta_p1, [x, y, z], post[m])
                        else:
                            f_m = basis_func(0, alpha_c, beta_c, [x, y, z], post[m])

                        if n == 0:
                            f_n = basis_func(0, alpha_p1, beta_p1, [x, y, z], post[n])
                        else:
                            f_n = basis_func(0, alpha_c, beta_c, [x, y, z], post[n])

                        c_m = evec1.T[1, m]
                        c_n = evec1.T[1, n]
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


dist1, point1, size_val1 = p1_3D(fid, cutoff, xrange, yrange, zrange, post, alpha_c, beta_c, alpha_p1, beta_p1, evec1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

co1 = dist1*(1/np.max(dist1))
ax.scatter(point1[:,0], point1[:,1], point1[:,2], c=co1, s=size_val1, alpha=0.2, cmap='viridis')

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")
ax.set_zlabel("z Position ($a_{0}$)")

ax.azim = 100
ax.elev = 8

''' Pyrrole 3D Density '''
fid = 60
cutoff = 0.06

xrange = np.linspace(-2, 2, fid)
yrange = np.linspace(-2, 2, fid)
zrange = np.linspace(-2, 2, fid)

post = [(1., 0., 0.), (1 / 2, np.sqrt(3) / 2, 0.), (-1 / 2, np.sqrt(3) / 2, 0.), (-1., 0., 0.),
        (-1 / 2, -np.sqrt(3) / 2, 0.), (1 / 2, -np.sqrt(3) / 2, 0.)]  # Benzene


@numba.njit
def p2_3D(fid, cutoff, xrange, yrange, zrange, post, alpha_c, beta_c, alpha_p2, beta_p2, evec2):
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
                        if m == 0:
                            f_m = basis_func(0, alpha_p2, beta_p2, [x, y, z], post[m])
                        else:
                            f_m = basis_func(0, alpha_c, beta_c, [x, y, z], post[m])

                        if n == 0:
                            f_n = basis_func(0, alpha_p2, beta_p2, [x, y, z], post[n])
                        else:
                            f_n = basis_func(0, alpha_c, beta_c, [x, y, z], post[n])

                        c_m = evec2.T[0, m]
                        c_n = evec2.T[0, n]
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


dist2, point2, size_val2 = p2_3D(fid, cutoff, xrange, yrange, zrange, post, alpha_c, beta_c, alpha_p2, beta_p2, evec2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

co2 = dist2*(1/np.max(dist2))
ax.scatter(point2[:,0], point2[:,1], point2[:,2], c=co2, s=size_val2, alpha=0.2, cmap='viridis')

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")
ax.set_zlabel("z Position ($a_{0}$)")

ax.azim = 100
ax.elev = 10
plt.show()