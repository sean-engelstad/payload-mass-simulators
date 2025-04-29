import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from _TS_elem import *

# aluminum beam
E = 70e9 # Pa
nu = 0.3 
G = E / 2.0 / (1+nu)
rho = 2.7e3

# geometry
L = 1.0
t1 = 1e-2
t2 = 1e-3
A = t1 * t2
Iy = t1**3 * t2 / 12.0
Iz = t2**3 * t1 / 12.0
ks = 5/6 # shear correction of rectangular cross-section
J = Iy + Iz
ky = ks
kz = ks

# analytic eigenvalues
# --------------------

omega_EB = np.array([1.875, 4.694, 7.855])**2 * np.sqrt(E * Iz / rho / A / L**4)
TS_denom = 1.0 + np.array([1,2,3])**2 * np.pi**2 * E * t1**2 / ks / G / L**2
omega_TS = omega_EB / np.sqrt(TS_denom)
print(f"{omega_EB=}\n{omega_TS=}")


# FEM modal analysis eigenvalues
# ------------------------------

# first get constitutive datums
CK = get_CK(E, A, G, J, Iy, Iz, ky, kz)
CM = get_CM(rho, A, Iy, Iz)
print(f"{CM=}")

# then build element stiffness and mass matrices (same for each element, so do up front)
nelems = 10
Le = L / nelems
# make a single xpts and get Kelem (will be same Kelem for all)
xpts = np.array([0.0] * 3 + [Le, 0.0, 0.0])
qvars = np.array([0.0]*12) # since linear static formulation, can just diff about 0
ref_axis = np.array([0.0, 1.0, 0.0])
Kelem = get_stiffness_matrix(xpts, qvars, ref_axis, CK)
Melem = get_mass_matrix(xpts, qvars, ref_axis, CM)

# print(f"{np.diag(Kelem)=}")
# plt.imshow(Kelem)
plt.imshow(Melem)
plt.show()

# get only w,thy DOF (to only show bending)
keep_elem_dof = np.array([1, 5, 7, 11])
Kelem = Kelem[keep_elem_dof,:][:,keep_elem_dof]
Melem = Melem[keep_elem_dof,:][:,keep_elem_dof]

# plt.imshow(np.log(Melem+1e-8))
# plt.show()

# now assemble a mass and stiffness matrix

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

# debug
# plt.imshow(Kr)
# plt.show()

plot_mode = False
if plot_mode:
    plt.figure()
    for imode in range(3):
        wvec = eigvecs[:,imode][::2]
        xvec = np.linspace(0.0, 1.0, wvec.shape[0])
        plt.plot(xvec, wvec, label=str(imode))
    plt.legend()
    plt.show()