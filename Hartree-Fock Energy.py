import math
import numpy as np
import scipy as sp
import numba

''' Molecular Information '''
# Define the information of the molecule
R = 1.98
alpha = 0.001
beta = 1.9

# Location, Atomic Number, No. Electrons (decimal for shared), No. Basis functions
atoms = [[(-R/2, 0., 0.), 1., 0.5, 18], [(R/2, 0., 0.), 1., 0.5, 18]] # Atomic no is protons
#atoms = [[(0., 0., 0.), 4, 4, 20]] # Atomic no is protons

# Size of the matrices
size = 0
for A in atoms:
    size += A[3]

# Total number of electrons
no_elec = 0
for A in atoms:
    no_elec += A[2]

# Create labelling system for use in further calculations
atoms_new = [] # Atoms including their indices in the matrix
prev = [] # How many indices previously calculated
for i in range(len(atoms)):
    i_range = 0
    for j in range(i):
        i_range += atoms[j][3]
    atoms_new.append([(i_range, i_range+atoms[i][3]-1), atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]])

print(atoms_new)

''' Basic Functions '''
@numba.njit
def basis_exp(alpha, beta, i):
    return alpha*(beta**(i))

@numba.njit
def R_AB(Apos, Bpos, i, j, alpha, beta):
    # A, B are position vectors
    li = basis_exp(alpha, beta, i)
    lj = basis_exp(alpha, beta, j)
    dist = (Apos[0]-Bpos[0])**2 + (Apos[1]-Bpos[1])**2 + (Apos[2]-Bpos[2])**2
    return np.exp(-((li*lj)/(li+lj))*dist)

@numba.njit
def v(Apos, Bpos, CI, i, j, alpha, beta):
    # A, B, CI are position vectors
    li = basis_exp(alpha, beta, i)
    lj = basis_exp(alpha, beta, j)
    P = [(li*Apos[0] + lj*Bpos[0])/(li+lj), (li*Apos[1] + lj*Bpos[1])/(li+lj), (li*Apos[2] + lj*Bpos[2])/(li+lj)]
    dist = (P[0]-CI[0])**2 + (P[1]-CI[1])**2 + (P[2]-CI[2])**2
    return (li + lj)*dist

@numba.njit
def v_ijkl(Apos, Bpos, Cpos, Dpos, i, j, k, l, alpha, beta):
    #A, B, C, D are position vectors (MUST BE LEFT TO RIGHT)
    li = basis_exp(alpha, beta, i)
    lj = basis_exp(alpha, beta, j)
    lk = basis_exp(alpha, beta, k)
    ll = basis_exp(alpha, beta, l)

    lijkl = li + lj + lk + ll

    P = [(li*Apos[0] + lj*Bpos[0])/(li+lj), (li*Apos[1] + lj*Bpos[1])/(li+lj), (li*Apos[2] + lj*Bpos[2])/(li+lj)]
    Q = [(lk*Cpos[0] + ll*Dpos[0])/(lk + ll), (lk*Cpos[1] + ll*Dpos[1])/(lk + ll), (lk*Cpos[2] + ll*Dpos[2])/(lk + ll)]

    dist = (P[0]-Q[0])**2 + (P[1]-Q[1])**2 + (P[2]-Q[2])**2
    return ((li + lj)*(lk + ll)*dist)/(lijkl)

@numba.njit
def error_v(v):
    if v != 0:
        return math.erf(np.sqrt(v))/np.sqrt(v)
    else:
        return 2/np.sqrt(np.pi)

''' Simplification Matrices '''
def lam_vec(size, atoms_new, alpha, beta):
    lvec = np.zeros((size))

    for A in atoms_new:
        for i in range(A[0][0], A[0][1] + 1):
            Ai = i - A[0][0]
            li = basis_exp(alpha, beta, Ai)
            lvec[i] = li

    return lvec


def R_AB_mat(atoms_new, size):
    RAB = np.zeros((size, size))

    for A in atoms_new:
        for B in atoms_new:
            dist = (A[1][0] - B[1][0]) ** 2 + (A[1][1] - B[1][1]) ** 2 + (A[1][2] - B[1][2]) ** 2
            for i in range(A[0][0], A[0][1] + 1):
                for j in range(B[0][0], B[0][1] + 1):
                    # i,j are positions in matrix s_mat; Ai, Bj are basis function indices for atoms A, B
                    Ai = i - A[0][0]
                    Bj = j - B[0][0]

                    li = basis_exp(alpha, beta, Ai)
                    lj = basis_exp(alpha, beta, Bj)

                    RAB[i, j] = np.exp(-((li * lj) / (li + lj)) * dist)

    return RAB


def t_mat_cons(atoms_new, size):
    t_cons = np.zeros((size, size))

    for A in atoms_new:
        for B in atoms_new:
            dist = (A[1][0] - B[1][0]) ** 2 + (A[1][1] - B[1][1]) ** 2 + (A[1][2] - B[1][2]) ** 2
            for i in range(A[0][0], A[0][1] + 1):
                for j in range(B[0][0], B[0][1] + 1):
                    # i,j are positions in matrix s_mat; Ai, Bj are basis function indices for atoms A, B
                    Ai = i - A[0][0]
                    Bj = j - B[0][0]

                    li = basis_exp(alpha, beta, Ai)
                    lj = basis_exp(alpha, beta, Bj)

                    t_cons[i, j] = 1 - ((2 * li * lj) / (3 * (li + lj))) * dist

    return t_cons


def v_mat_cons(atoms_new, size):
    v_cons = np.zeros((size, size))

    for A in atoms_new:
        for B in atoms_new:
            for i in range(A[0][0], A[0][1] + 1):
                for j in range(B[0][0], B[0][1] + 1):
                    # i,j are positions in matrix s_mat; Ai, Bj are basis function indices for atoms A, B
                    Ai = i - A[0][0]
                    Bj = j - B[0][0]

                    error_cal = 0
                    for C in atoms_new:
                        v_C = v(A[1], B[1], C[1], Ai, Bj, alpha, beta)
                        error_cal += C[2] * error_v(v_C)

                    v_cons[i, j] = error_cal

    return v_cons

''' Hartree Matrices '''


def S_matrix(lvec, RAB):
    lcons = np.divide(np.power(4 * np.outer(lvec, lvec), 3 / 4), np.power(
        np.repeat(lvec[:, np.newaxis], len(lvec), axis=1) + np.repeat(lvec[:, np.newaxis], len(lvec), axis=1).T, 3 / 2))

    s_mat = lcons * RAB

    return s_mat


def T_matrix(lvec, RAB, t_cons):
    lcons = (3 / 4) * np.divide(np.power(4 * np.outer(lvec, lvec), 7 / 4), np.power(
        np.repeat(lvec[:, np.newaxis], len(lvec), axis=1) + np.repeat(lvec[:, np.newaxis], len(lvec), axis=1).T, 5 / 2))

    t_mat = lcons * RAB * t_cons

    return t_mat


def V_matrix(lvec, RAB, v_cons):
    lcons = -np.divide(np.power(4 * np.outer(lvec, lvec), 3 / 4),
                       np.repeat(lvec[:, np.newaxis], len(lvec), axis=1) + np.repeat(lvec[:, np.newaxis], len(lvec),
                                                                                     axis=1).T)

    v_mat = lcons * RAB * v_cons

    return v_mat


@numba.njit
def J_matrix(size, a_range, a_pos, D_new, alpha, beta):
    j_mat = np.zeros((size, size))

    # Loop to get correct matrix indices
    for Aind in range(len(a_range)):
        for Bind in range(len(a_range)):
            for i in range(a_range[Aind][0], a_range[Aind][1] + 1):
                Ai = i - a_range[Aind][0]

                li = basis_exp(alpha, beta, Ai)
                for j in range(a_range[Bind][0], a_range[Bind][1] + 1):
                    # i,j are positions in matrix s_mat; Ai, Bj are basis function indices for atoms A, B
                    Bj = j - a_range[Bind][0]

                    lj = basis_exp(alpha, beta, Bj)
                    RAB = R_AB(a_pos[Aind], a_pos[Bind], Ai, Bj, alpha, beta)

                    # Loop to consider all interactions
                    for Cind in range(len(a_range)):
                        for Dind in range(len(a_range)):
                            for k in range(a_range[Cind][0], a_range[Cind][1] + 1):
                                Ck = k - a_range[Cind][0]

                                lk = basis_exp(alpha, beta, Ck)
                                for l in range(a_range[Dind][0], a_range[Dind][1] + 1):
                                    # k,l are positions in matrix s_mat; Ck, Dl are basis function indices for atoms C, D
                                    Dl = l - a_range[Dind][0]

                                    ll = basis_exp(alpha, beta, Dl)
                                    lijkl = li + lj + lk + ll
                                    RCD = R_AB(a_pos[Cind], a_pos[Dind], Ck, Dl, alpha, beta)
                                    vijkl = v_ijkl(a_pos[Aind], a_pos[Bind], a_pos[Cind], a_pos[Dind], Ai, Bj, Ck, Dl,
                                                   alpha, beta)
                                    error_ijkl = error_v(vijkl)

                                    j_mat[i, j] += 8 * D_new[k, l] * RAB * RCD * (((li * lj * lk * ll) ** (3 / 4)) / (
                                                (li + lj) * (lk + ll) * np.sqrt(lijkl))) * error_ijkl

    return j_mat

@numba.njit
def K_matrix(size, a_range, a_pos, D_new, alpha, beta):
    k_mat = np.zeros((size, size))

    # Loop to get correct matrix indices
    for Aind in range(len(a_range)):
        for Bind in range(len(a_range)):
            for i in range(a_range[Aind][0], a_range[Aind][1] + 1):
                Ai = i - a_range[Aind][0]

                li = basis_exp(alpha, beta, Ai)
                for j in range(a_range[Bind][0], a_range[Bind][1] + 1):
                    # i,j are positions in matrix s_mat; Ai, Bj are basis function indices for atoms A, B
                    Bj = j - a_range[Bind][0]

                    lj = basis_exp(alpha, beta, Bj)

                    # Loop to consider all interactions
                    for Cind in range(len(a_range)):
                        for Dind in range(len(a_range)):
                            for k in range(a_range[Cind][0], a_range[Cind][1] + 1):
                                Ck = k - a_range[Cind][0]

                                lk = basis_exp(alpha, beta, Ck)
                                for l in range(a_range[Dind][0], a_range[Dind][1] + 1):
                                    # k,l are positions in matrix s_mat; Ck, Dl are basis function indices for atoms C, D
                                    Dl = l - a_range[Dind][0]

                                    ll = basis_exp(alpha, beta, Dl)
                                    lijkl = li + lj + lk + ll
                                    vilkj = v_ijkl(a_pos[Aind], a_pos[Dind], a_pos[Cind], a_pos[Bind], Ai, Dl, Ck, Bj,
                                                   alpha, beta)
                                    error_ilkj = error_v(vilkj)

                                    RAD = R_AB(a_pos[Aind], a_pos[Dind], Ai, Dl, alpha, beta)
                                    RCB = R_AB(a_pos[Cind], a_pos[Bind], Ck, Bj, alpha, beta)

                                    k_mat[i, j] += 8 * D_new[k, l] * RAD * RCB * (((li * lj * lk * ll) ** (3 / 4)) / (
                                                (li + ll) * (lk + lj) * np.sqrt(lijkl))) * error_ilkj

    return k_mat

''' Update Matrices '''


@numba.njit
def density_matrix(evec, size, no_elec):
    D_new = np.zeros((size, size))
    for k in range(size):
        for l in range(size):
            for ne in range(int(no_elec / 2)):
                D_new[k, l] += 2 * (evec[k, ne] * evec[l, ne])
    return D_new


def total_energy(atoms_new, D, t_mat, v_mat, g_mat, size):
    Enuc = 0
    for I in atoms_new:
        for J in atoms_new:
            if I[1] != J[1]:
                dist = np.sqrt((I[1][0] - J[1][0]) ** 2 + (I[1][1] - J[1][1]) ** 2 + (I[1][2] - J[1][2]) ** 2)
                Enuc += 0.5 * (I[2] * J[2]) / (dist)

    E1 = 0
    for i in range(size):
        for j in range(size):
            E1 += (t_mat[i, j] + v_mat[i, j]) * D[i, j]

    E2 = 0
    for i in range(size):
        for j in range(size):
            E2 += g_mat[i, j] * D[i, j]
    E2 = 0.5 * E2

    return Enuc + E1 + E2


@numba.njit
def energy_difference(E_new, E_old):
    return np.abs(E_old - E_new) / (np.abs(E_new) + 1)


@numba.njit
def density_difference(D_new, D_old, size):
    D_diff = 0
    for i in range(size):
        for j in range(size):
            D_diff += np.abs(D_new[i, j] - D_old[i, j])
    D_diff = (1 / size ** 2) * D_diff
    return D_diff

''' Calculation '''
''' Perform the Molecular Hartree Fock Calculations '''
E_bound = 1e-10
D_bound = 1e-10

E_new = 10
D_new = np.zeros((size, size))

E_diff = 1
D_diff = 1

count = -1

# Define constant matrix constants
lvec = lam_vec(size, atoms_new, alpha, beta)
RAB = R_AB_mat(atoms_new, size)
t_cons = t_mat_cons(atoms_new, size)
v_cons = v_mat_cons(atoms_new, size)

# Define the constant matrices
s_mat = S_matrix(lvec, RAB)
t_mat = T_matrix(lvec, RAB, t_cons)
v_mat = V_matrix(lvec, RAB, v_cons)

# Define varying matrix constants
a_range = []
a_pos = []
for A in atoms_new:
    a_range.append(A[0])
    a_pos.append(A[1])

while E_diff > E_bound or D_diff > D_bound:
    count += 1

    # Save previous results for comparison
    E_old = E_new
    D_old = D_new

    # Create new matrices
    j_mat = J_matrix(size, a_range, a_pos, D_new, alpha, beta)
    k_mat = K_matrix(size, a_range, a_pos, D_new, alpha, beta)
    g_mat = j_mat - 0.5 * k_mat

    f_mat = t_mat + v_mat + g_mat

    # Solve the eigenvalue equation
    eval, evec = sp.linalg.eigh(f_mat, s_mat)

    # Calculate new total energy
    E_new = total_energy(atoms_new, D_new, t_mat, v_mat, g_mat, size)

    # Calculate new density matrix
    D_new = density_matrix(evec, size, no_elec)

    E_diff = energy_difference(E_new, E_old)
    D_diff = density_difference(D_new, D_old, size)

    print('')
    print(E_diff)
    print(D_diff)

print('')
print(count)
print(eval[0])
print(E_new)