import numpy as np
import scipy as sp

import adaptive

import matplotlib.pyplot as plt

''' Adaptive Calculation of Radial Energy '''
fname = 'H2.learner'

def test(R):
    return R

learner = adaptive.Learner1D(test, (0., 0.))
learner.load(fname)
data = learner.to_numpy()
print(data)

''' Interpolate Data '''
def LJ_simp(r, epsilon, sigma):
    return 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

def LJ_gen(r, A, B, n, m):
    return A/(r**n) - B/(r**m)

popt, _ = sp.optimize.curve_fit(LJ_simp, data[:,0], data[:,1], maxfev=10000)
epsilon_simp, sigma_simp = popt

popt, _ = sp.optimize.curve_fit(LJ_gen, data[:,0], data[:,1], maxfev=10000)
A_gen, B_gen, n_gen, m_gen = popt

''' Plot Radial Dependence '''
fig, ax = plt.subplots()

# Plot Simulated Data
line1, = ax.plot(data[:,0], data[:,1], color='b')
ax.axvline(data[np.argmin(data[:,1]),0], color='y', linestyle='--')
ax.text(data[np.argmin(data[:,1]),0]+0.05, 6, "(" + str(np.round(data[np.argmin(data[:,1]),0], 5)) + "," + str(np.round(data[np.argmin(data[:,1]),1], 5)) + ")")

# Plot Fitted Data
line2, = ax.plot(data[:,0], LJ_simp(data[:,0], epsilon_simp, sigma_simp), color='g')
line3, = ax.plot(data[:,0], LJ_gen(data[:,0], A_gen, B_gen, n_gen, m_gen), color='r')

ax.set_xlabel("Radius ($a_{0}$)")
ax.set_ylabel("Total Energy ($E_{h}$)")

# Add Legend
ax.legend([line1, line2, line3], ['Hartree-Fock', '12-6 Potential', 'General Potential'])

plt.show()