import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from collections import Counter

# Number of Carbon Atoms
Nc = 60
# 2pz orbital energy of carbon
alpha = -0.6
# Nearest-neighbour interaction energy
beta = -0.1

''' Create the Overlap/Interaction Matrices '''
''' BuckminsterFullerene (Nc=60) '''
# F Matrix
F = np.zeros((Nc, Nc))
# S Matrix
S = np.zeros((Nc, Nc))

for i in range(Nc):
    for j in range(Nc):
        if i%(Nc) == j%(Nc):
            F[i,j] = alpha
            S[i,j] = 1
        elif abs(i-j) == 1:
            F[i,j] = beta
# Manual Connection for first point in each level
F[0, 5] = beta
F[5, 0] = beta
F[6, 20] = beta
F[20, 6] = beta
F[21, 38] = beta
F[38, 21] = beta
F[39, 53] = beta
F[53, 39] = beta
F[54, 59] = beta
F[59, 54] = beta

# Manual Vertical Connections from 0th to 1st layer
F[0, 8] = beta
F[8, 0] = beta
F[1, 11] = beta
F[11, 1] = beta
F[2, 13] = beta
F[13, 2] = beta
F[3, 16] = beta
F[16, 3] = beta
F[4, 18] = beta
F[18, 4] = beta

# Manual Vertical Connections from 1st to 2nd layer
F[7, 23] = beta
F[23, 7] = beta
F[9, 25] = beta
F[25, 9] = beta
F[10, 27] = beta
F[27, 10] = beta
F[12, 29] = beta
F[29, 12] = beta
F[14, 31] = beta
F[31, 14] = beta
F[15, 33] = beta
F[33, 15] = beta
F[17, 35] = beta
F[35, 17] = beta
F[19, 37] = beta
F[37, 19] = beta

# Manual Vertical Connections from 2nd to 3rd layer
F[22, 41] = beta
F[41, 22] = beta
F[24, 42] = beta
F[42, 24] = beta
F[26, 44] = beta
F[44, 26] = beta
F[28, 46] = beta
F[46, 28] = beta
F[30, 47] = beta
F[47, 30] = beta
F[32, 49] = beta
F[49, 32] = beta
F[34, 51] = beta
F[51, 34] = beta
F[36, 52] = beta
F[52, 36] = beta

# Manual Vertical Connections from 3rd to 4th layer
F[40, 55] = beta
F[55, 40] = beta
F[43, 56] = beta
F[56, 43] = beta
F[45, 57] = beta
F[57, 45] = beta
F[48, 58] = beta
F[58, 48] = beta
F[50, 59] = beta
F[59, 50] = beta

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
