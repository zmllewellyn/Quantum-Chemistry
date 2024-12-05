import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from collections import Counter

# Number of Carbon Atoms
Nc = 2
# 2pz orbital energy of carbon
alpha = -0.6
# Nearest-neighbour interaction energy
beta = -0.1

''' Create the Overlap/Interaction Matrices '''
# F Matrix
F = np.zeros((Nc, Nc))
# S Matrix
S = np.zeros((Nc, Nc))
for i in range(Nc):
    for j in range(Nc):
        if i%Nc == j%Nc:
            F[i,j] = alpha
            S[i,j] = 1
        elif abs(i-j) == 1 or abs(i-j) == Nc-1:
            F[i,j] = beta

''' Get eigenvalues/eigenvectors '''
sol = np.linalg.solve(S, F)
eval, evec = np.linalg.eigh(sol)

''' Plot Energy Levels '''
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

evals_degen = Counter(np.round(eval, decimals=6))
max_degen = 0
for ev in list(evals_degen.keys()):
    ax.barh(ev, height=0.005, width=evals_degen[ev], color='r')

    if evals_degen[ev] > max_degen:
        max_degen = evals_degen[ev]

ax.set_xlabel("Degeneracy")
ax.set_ylabel("Energy Level ($E_{h}$)")
ax.set_xticks(np.arange(1, max_degen + 1))

plt.show()

''' Create the basis functions '''
def lam(i, alpha, beta):
    return alpha*(beta**(i-1))

def basis_func(i, alpha, beta, pos, atom_pos):
    r = np.sqrt((pos[0]-atom_pos[0])**2 + (pos[1]-atom_pos[1])**2 + (pos[2]-atom_pos[2])**2)
    theta = np.arctan2(np.sqrt((pos[0]-atom_pos[0])**2 + (pos[1]-atom_pos[1])**2),pos[2])
    phi = np.arctan2((pos[1]-atom_pos[1]), (pos[0]-atom_pos[0]))
    li = 2.
    return 0.5*np.sqrt(3/np.pi)*np.sqrt(((2*li)**5)/(4*3*2*1))*r*np.exp(-li*r)*np.cos(theta)

''' Create/Plot the 3D Electron Density'''
fid = 20

xrange = np.linspace(-2, 2, fid)
yrange = np.linspace(-2, 2, fid)
zrange = np.linspace(-2, 2, fid)

post = [(0,-0.405,0), (0,0.405,0)] # Ethene
dist = np.zeros((fid, fid, fid))
size_val = np.zeros((fid, fid, fid))
cutoff = 0.02
e_level = 0 # Energy level

for i in range(fid):
    for j in range(fid):
        for k in range(fid):
            x = xrange[i]
            y = yrange[j]
            z = zrange[k]

            for m in range(len(post)):
                for n in range(len(post)):
                    f_m = basis_func(0, alpha, beta, [x,y,z], post[m])
                    f_n = basis_func(0, alpha, beta, [x,y,z], post[n])
                    c_m = evec.T[e_level,m]
                    c_n = evec.T[e_level,n]
                    dist[i,j,k] += c_m*c_n*f_m*f_n

            if dist[i,j,k] <= cutoff:
                size_val[i,j,k] = 0
            else:
                size_val[i,j,k] = 20

point = []
for i in range(fid):
    for j in range(fid):
        for k in range(fid):
            x = xrange[i]
            y = yrange[j]
            z = zrange[k]

            point.append([x,y,z])

dist = np.round(dist,5)
point = np.asarray(point)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

co = dist*(1/np.max(dist))
ax.scatter(point[:,0], point[:,1], point[:,2], c=co, s=size_val, alpha=0.2, cmap='viridis')

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")
ax.set_zlabel("z Position ($a_{0}$)")

ax.azim = 150
ax.elev = 10
plt.show()

''' Calculate and Plot the Cross-sections'''
fid = 120
z = 0.2
xrange = np.linspace(-2, 2, fid)
yrange = np.linspace(-2, 2, fid)

post = [(0,-0.405,0), (0,0.405,0)] # Ethene

dist = np.zeros((fid, fid))

for i in range(fid):
    for j in range(fid):
        x = xrange[i]
        y = yrange[j]

        for m in range(len(post)):
            for n in range(len(post)):
                f_m = basis_func(0, alpha, beta, [x, y, z], post[m])
                f_n = basis_func(0, alpha, beta, [x, y, z], post[n])
                c_m = evec.T[0, m]
                c_n = evec.T[0, n]
                dist[i, j] += c_m * c_n * f_m * f_n

dist = np.round(dist, 5).T

fig, ax = plt.subplots()

vmin = np.min(dist)
vmax = np.max(dist)
cmap = cm.viridis
norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

ax.pcolormesh(xrange, yrange, dist)

ax.set_xlabel("x Position ($a_{0}$)")
ax.set_ylabel("y Position ($a_{0}$)")

ax.set_title("n=2, z=0.5 Cross-section")

fig.colorbar(sm, ax=ax, label="Electron Density $|\\Psi|^{2}$")
plt.show()