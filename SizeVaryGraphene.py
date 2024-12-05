import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from collections import Counter

# 2pz orbital energy of carbon
alpha = -0.6
# Nearest-neighbour interaction energy
beta = -0.1

''' Create the Overlap/Interaction Matrices '''
''' Graphene Rings '''
# How many rings deep
depth = 1
# The total number of carbon atoms is
Nc = 6*(depth**2)
# F Matrix
F = np.zeros((Nc, Nc))
# S Matrix
S = np.zeros((Nc, Nc))

ring_index = [0]
for d in range(0,depth):
    val = 6*(d+1)**2
    ring_index.append(val)

info = np.zeros(Nc)

# Basic Path
for i in range(Nc):
    for j in range(Nc):
        if i%Nc == j%Nc:
            F[i,j] = alpha
            S[i,j] = 1
        elif abs(i-j) == 1:
            F[i,j] = beta
            info[i] += 1

# Fill in all gaps in the rings
for ind in range(1,len(ring_index)):
    F[ring_index[ind]-1,ring_index[ind-1]] = beta
    F[ring_index[ind-1],ring_index[ind]-1] = beta
    info[ring_index[ind]-1] += 1
    info[ring_index[ind-1]] += 1

# Fill in radial components
for i in range(2, len(ring_index)):
    hop = (i-2)*[2] + [3]
    count = ring_index[i-1]
    hop_index = 0
    count += hop[hop_index]
    while count < ring_index[i]:
        low_unfill = 0
        for j in range(len(info)):
            if info[j] == 2:
                low_unfill = j
                break
        F[count,low_unfill] = beta
        F[low_unfill,count] = beta
        info[count] += 1
        info[low_unfill] +=1
        hop_index = (hop_index+1)%len(hop)
        count += hop[hop_index]

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