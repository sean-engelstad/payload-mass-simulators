# now let's test this out and visualize it
import sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
sys.path.append("beam-fea")
from beam_elem import *
from scipy.linalg import eigh

# axial torsion, and transverse beam oriented in x direction
# so y,z are transverse directions

# assume each node has 6 DOF (although of course in this case)
# here are the material properties

rho = 2.7e3 # kg/m^3
E = 70e9 # aluminum (Pa)
nu = 0.3
G = E / 2.0 / (1 + nu)
ty = 0.005 # thicknesses (m)
tz = 0.007 
A = ty * tz
Iy = tz**3 * ty / 12.0 # rotation in xz plane
Iz = ty**3 * tz / 12.0
J = Iy + Iz # polar moment of inertia
L = 1 # m

# construct the global stiffness matrix
# -------------------------------------
dof_per_node = 6
nnodes = 2
ndof = nnodes * dof_per_node
K = np.zeros((ndof, ndof))
# dof per node are [u,v,w,thx,thy,thz]

# get element stiffness matrices
xscale = 1.0 # length of beam
Kx_ax = get_kelem_axial(E, A, L)
Kx_tor = get_kelem_torsion(G, J, L)
Kz_tr = get_kelem_transverse(xscale, E, Iy)
Ky_tr = get_kelem_transverse(xscale, E, Iz)

# print(f"{Kx_ax=}")
# print(f"{Kx_tor=}")
# print(f"{J/L=} {A=}")

# assemble the global stiffness (linear elasticity)
axial_dof = [0,6]
ax_rows, ax_cols = np.ix_(axial_dof, axial_dof)
K[ax_rows, ax_cols] += Kx_ax

# torsion
torsion_dof = [3,9]
tor_rows, tor_cols = np.ix_(torsion_dof, torsion_dof) # thx spots
K[tor_rows, tor_cols] += Kx_tor

# bending xy
ybending_dof = [1,5,7,11]
y_rows, y_cols = np.ix_(ybending_dof, ybending_dof)
K[y_rows, y_cols] += Ky_tr

# bending xz
zbending_dof = [2,4,8,10]
z_rows, z_cols = np.ix_(zbending_dof, zbending_dof)
K[z_rows, z_cols] += Kz_tr

# plt.imshow(np.log(1e-5 + K))
# plt.show()

# print(f"{K=}")

# now assemble mass matrix
# ---------------------------------
M = np.zeros((ndof, ndof))

Mx_ax = get_melem_axial(rho, A, L)
Mx_tor = get_melem_torsion(rho, A, J, L)
Mz_tr = get_melem_transverse(xscale, rho, A)
My_tr = get_melem_transverse(xscale, rho, A)

# print(f"{Mx_tor=}")
# torsional mass is incredibly low (very high torsional frequencies, so never show up)
# Mx_tor *= 1e4

# now apply a tip load and some root boundary conditions
M[ax_rows, ax_cols] += Mx_ax
M[tor_rows, tor_cols] += Mx_tor
M[y_rows, y_cols] += My_tr
M[z_rows, z_cols] += Mz_tr

# scale up mass matrix to avoid numerical issues?
# M *= 1e6

# plt.imshow(np.log(1e-5 + M))
# plt.show()

# apply bcs to get reduced matrix
# ------------------------------------------

# fix u,v,w,thx,thy,thz at 0
bcs = [_ for _ in range(6)]

# K[bcs,:] = 0.0
# K[:,bcs] = 0.0
# for bc in bcs:
#     K[bc,:] = 0.0
#     K[:,bc] = 0.0
#     K[bc,bc] = 1.0

# no we actually need to get reduced stiffness and mass matrices
# otherwise non-zero eigen displacements can occur at BCs which doesn't make sense
keep_dof = [_ for _ in range(ndof) if not(_ in bcs)]
Kr = K[keep_dof,:][:,keep_dof]
Mr = M[keep_dof,:][:,keep_dof]

keep_axial_dof = [i for i,_ in enumerate(keep_dof) if _ in axial_dof]
keep_ybending = [i for i,_ in enumerate(keep_dof) if _ in ybending_dof]
keep_zbending = [i for i,_ in enumerate(keep_dof) if _ in zbending_dof]
# print(f"{keep_zbending=}")
# exit()

# solve eigenvalue problem
# ------------------------

# solve symmetric eigenvalue problem
eigvals, eigvecs = eigh(Kr, Mr)
print(f"{eigvals=}")
plt.imshow(eigvecs[:,:5])
plt.show()

# exact beam bending (with no distributed loads) is 3rd order cubic polynomial
# so can back-solve the exact transverse eigenmodes from one beam element nodal disps
# in each transverse direction

# compute exact eigenmodes using cubic polynomials for transverse, linear for axial + torsion
# -------------------------------------------------------------
# a0 + a1 * eta + a2 * eta^2 + a3 * eta^3 = u(eta)
# th(eta) = a1 + 2 * a2 * eta + 3 * a3 * eta^2


def tr_get_acoeff(vals):
    u0 = vals[0]; u1 = vals[1]; th0 = vals[2]; th1 = vals[3]
    a = [0]*4
    a[0] = u0
    a[1] = th0
    a[2] = 3 * u0 - 3 * u1 - 2 * th0 + th1
    a[3] = 2 * (u0 - u1) + th0 + th1
    return a

def eval_a_poly(avec, eta_vec):
    return avec[0] + avec[1] * eta_vec + avec[2] * eta_vec**2 + avec[3] * eta_vec**3

fig, axs = plt.subplots(4, 1, figsize=(10,6))
eta = np.linspace(0.0, 1.0, 30)
for imode in range(4):
    phi = eigvecs[:,imode]
    # normalize the mode

    ax_phi = [0, phi[0]]
    # tor_phi = phi[torsion_dof]
    ytr_phi = [0, phi[1], 0.0, phi[5]]
    ztr_phi = [0, phi[2], 0.0, phi[4]]

    # plot axial part of mode
    axs[imode].plot([0,1], ax_phi, label=f"u-{imode}")
    
    # plot y transverse part of mode
    a_ytr = tr_get_acoeff(ytr_phi)
    v = eval_a_poly(a_ytr, eta)
    axs[imode].plot(eta, v, label=f"v-{imode}")

    # plot z transverse part of mode
    a_ztr = tr_get_acoeff(ztr_phi)
    w = eval_a_poly(a_ztr, eta)
    axs[imode].plot(eta, w, label=f"w-{imode}")

plt.legend()
plt.show()

# can use this to then write out the solution to a VTK file
# ---------------------------------------------------------