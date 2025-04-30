import numpy as np
import matplotlib.pyplot as plt
from _elems import TS_element_stiffness, TS_element_mass, TS_distr_load
from scipy.linalg import eigh

# prelim problem constants
# ------------------------

# aluminum beam
E = 70e9 # Pa
nu = 0.3 
G = E / 2.0 / (1+nu)
rho = 1.225e3

# geometry
L = 1.0
hL = 1e-1
t1 = hL * L
t2 = 1e-2
A = t1 * t2
I = t1**3 * t2 / 12.0
ks = 5/6 # shear correction of rectangular cross-section

# distributed load 
q = 1.0

# analytic solution
# -----------------

# euler-bernoulli center plate deflection
w_EB = 5 * q * L**4 / 384.0 / E / I
# I think the analytic for timoshenko is kind of crap..

# FEM linear static problem
# -------------------------

# simply supported TS beam
nelems = 100
a = L / nelems / 2
Kelem = TS_element_stiffness(E, I, ks, G, A, a)
Felem = TS_distr_load(q, a)

print(f"{hL=} {I=} {A=} {a=}")

print(f"{np.diag(Kelem)=}")

nnodes = nelems+1
ndof = 2 * nnodes
K = np.zeros((ndof,ndof))
F = np.zeros((ndof,))
for i in range(nelems):
    glob_dof = np.array([2 * i, 2* i+1, 2 * i+2, 2*i+3])
    # print(f"{glob_dof=}")
    rows, cols = np.ix_(glob_dof, glob_dof)
    K[rows, cols] += Kelem
    F[glob_dof] += Felem[:,0]

remove_dof = [0, ndof-2]
keep_dof = [_ for _ in range(ndof) if not(_ in remove_dof)]
Kr = K[keep_dof,:][:,keep_dof]
Fr = F[keep_dof]
ur = np.linalg.solve(Kr, Fr)
ufull = np.zeros((ndof,))
ufull[keep_dof] = ur[:]

# max centerplate deflection
w_FEM = np.max(ufull[::2])
print(f"{w_EB=}\n{w_FEM=}")

# plot the solution
plot = False
if plot:
    plt.figure()
    wvec = ufull[::2]
    xvec = np.linspace(0.0, 1.0, wvec.shape[0])
    plt.plot(xvec, wvec)
    plt.show()

