import numpy as np
import matplotlib.pyplot as plt
# from _elems import TS_element_stiffness, TS_element_mass
from _elems2 import TS_element_stiffness, TS_element_mass
from scipy.linalg import eigh

# prelim problem constants
# ------------------------

# aluminum beam
E = 70e9 # Pa
nu = 0.3 
G = E / 2.0 / (1+nu)
rho = 1.225e3

# geometry
L = 1
t1 = 1e-1
t2 = 1e-2
A = t1 * t2
I = t1**3 * t2 / 12.0
ks = 5/6 # shear correction of rectangular cross-section

# analytic eigenvalues
# --------------------

omega_EB = np.array([1.875, 4.694, 7.855])**2 * np.sqrt(E * I / rho / A / L**4)
TS_denom = 1.0 + np.array([1,2,3])**2 * np.pi**2 * E * t1**2 / ks / G / L**2
omega_TS = omega_EB / np.sqrt(TS_denom)
print(f"{omega_EB=}\n{omega_TS=}")

# FEM modal analysis eigenvalues
# ------------------------------

nelems = 100
a = L / nelems
Kelem = TS_element_stiffness(a, E, I, ks, G, A)
Melem = TS_element_mass(a, rho, A, I)

print(F"{Kelem=}")

# plt.imshow(Melem)
# plt.show()

nnodes = nelems+1
ndof = 2 * nnodes
K = np.zeros((ndof,ndof))
M = np.zeros((ndof, ndof))
for i in range(nelems):
    glob_dof = np.array([2 * i, 2* i+1, 2 * i+2, 2*i+3])
    # print(f"{glob_dof=}")
    rows, cols = np.ix_(glob_dof, glob_dof)
    K[rows, cols] += Kelem
    M[rows, cols] += Melem

# apply bcs at root all w1, th1 are zero, with reduced operation
keep_dof = [_ for _ in range(2, ndof)]
# print(f"{keep_dof=}")
Kr = K[keep_dof,:][:,keep_dof]
Mr = M[keep_dof,:][:,keep_dof]

# solve the modal analysis
eigvals, eigvecs = eigh(Kr, Mr)
omega_FEM = np.sqrt(eigvals)
print(f"{omega_FEM[:3]=}")

# plt.figure()
# for imode in range(3):
#     wvec = eigvecs[:,imode][::2]
#     xvec = np.linspace(0.0, 1.0, wvec.shape[0])
#     plt.plot(xvec, wvec, label=str(imode))
# plt.legend()
# plt.show()
 
