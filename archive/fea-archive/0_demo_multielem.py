# now let's test this out and visualize it
import sys, numpy as np
import matplotlib.pyplot as plt
import niceplots
from matplotlib import colors
sys.path.append("beam-fea")
from _beam_elem import *
from scipy.linalg import eigh

# axial torsion, and transverse beam oriented in x direction
# so y,z are transverse directions

# assume each node has 6 DOF (although of course in this case)
# here are the material properties

rho = 2.7e3 # kg/m^3
E = 70e9 # aluminum (Pa)
nu = 0.3
G = E / 2.0 / (1 + nu)
ty = 0.01 # thicknesses (m)
tz = 0.01 
A = ty * tz
Iy = tz**3 * ty / 12.0 # rotation in xz plane
Iz = ty**3 * tz / 12.0
J = Iy + Iz # polar moment of inertia
L = 0.1 # m

# mesh connectivity and nodes
# -------------------------------------
nx = 11
nxe = nx - 1 # 10 elements

elem_conn = []
for ielem in range(nxe):
    elem_conn += [[ielem, ielem+1]]

nnodes = nx

# print(f'{elem_conn=}')
# exit()


# construct the global stiffness matrix
# -------------------------------------
dof_per_node = 6
ndof = nnodes * dof_per_node
K = np.zeros((ndof, ndof))
M = np.zeros((ndof, ndof))

# dof per node are [u,v,w,thx,thy,thz]

# get element stiffness matrices
xscale = L / nxe # length of beam
Kx_ax = get_kelem_axial(E, A, xscale)
Kx_tor = get_kelem_torsion(G, J, xscale)
Kz_tr = get_kelem_transverse(xscale, E, Iy)
Ky_tr = get_kelem_transverse(xscale, E, Iz)

# get element mass matrices
Mx_ax = get_melem_axial(rho, A, L)
Mx_tor = get_melem_torsion(rho, A, J, L)
Mz_tr = get_melem_transverse(xscale, rho, A)
My_tr = get_melem_transverse(xscale, rho, A)

axial_nodal_dof = [0]
torsion_nodal_dof = [3]
ytr_nodal_dof = [1,5]
ztr_nodal_dof = [2,4]

for ielem in range(nxe):
    nodes = elem_conn[ielem]
    # get global dof for each physics and this element
    axial_dof = [6*inode + _dof for _dof in axial_nodal_dof for inode in nodes]
    torsion_dof = [6*inode + _dof for _dof in torsion_nodal_dof for inode in nodes]
    ytr_dof = np.sort([6*inode + _dof for _dof in ytr_nodal_dof for inode in nodes])
    ztr_dof = np.sort([6*inode + _dof for _dof in ztr_nodal_dof for inode in nodes])

    # print(f"{ielem=} {ytr_dof=}")

    # get assembly arrays
    ax_rows, ax_cols = np.ix_(axial_dof, axial_dof)
    tor_rows, tor_cols = np.ix_(torsion_dof, torsion_dof) # thx spots
    y_rows, y_cols = np.ix_(ytr_dof, ytr_dof)
    z_rows, z_cols = np.ix_(ztr_dof, ztr_dof)

    # print(f'{axial_dof=} {torsion_dof=} {ytr_dof=} {ztr_dof=}')
    # exit()

    # assemble to global stiffness matrix
    K[ax_rows, ax_cols] += Kx_ax
    K[tor_rows, tor_cols] += Kx_tor
    K[y_rows, y_cols] += Ky_tr
    K[z_rows, z_cols] += Kz_tr

    # assemble to global mass matrix
    M[ax_rows, ax_cols] += Mx_ax
    M[tor_rows, tor_cols] += Mx_tor
    M[y_rows, y_cols] += My_tr
    M[z_rows, z_cols] += Mz_tr




# apply bcs to get reduced matrix
# ------------------------------------------

# fix u,v,w,thx,thy,thz at 0
bcs = [_ for _ in range(6)]

# no we actually need to get reduced stiffness and mass matrices
# otherwise non-zero eigen displacements can occur at BCs which doesn't make sense
keep_dof = [_ for _ in range(ndof) if not(_ in bcs)]
Kr = K[keep_dof,:][:,keep_dof]
Mr = M[keep_dof,:][:,keep_dof]

norm = colors.SymLogNorm(vmin=-10, vmax=10, linthresh=1e-4)
plt.imshow(Kr, norm=norm)
plt.colorbar()
plt.show()
print(f"{Kr=}")

keep_axial_dof = [i for i,_ in enumerate(keep_dof) if _ in axial_dof]
keep_ybending = [i for i,_ in enumerate(keep_dof) if _ in ytr_dof]
keep_zbending = [i for i,_ in enumerate(keep_dof) if _ in ztr_dof]

# solve eigenvalue problem
# ------------------------

# # change to just ytr temporarily for debugging
# ytr_global_dof = np.sort([6*inode+_dof for _dof in ytr_nodal_dof for inode in range(nx-1)])
# print(f"{ytr_global_dof=}")
# Kr = Kr[ytr_global_dof,:][:,ytr_global_dof]
# Mr = Mr[ytr_global_dof,:][:,ytr_global_dof]

# solve symmetric eigenvalue problem
eigvals, eigvecs = eigh(Kr, Mr)
freqs = np.sqrt(eigvals)
print(f"{freqs[:6]=}")
# plt.imshow(eigvecs[:,:5])
# plt.show()
# exit()

eta = np.linspace(0.0, 1.0, nx)

# then plot temporary ytr displacements only
# for imode in range(4):
#     phi_nobc = eigvecs[:,imode]
#     phi = [0] + list(phi_nobc[0::2])
#     plt.plot(eta, phi, label=f"mode{imode}")
# plt.legend()
# plt.show()

plt.style.use(niceplots.get_style())
disp_strs = ['u', 'v', 'w']
fig, axs = plt.subplots(4, 1, figsize=(10,10))
for imode in range(4):
    phi_r = eigvecs[:,imode]
    phi = np.zeros((ndof,))
    phi[keep_dof] = phi_r[:]
    
    for i in range(3):
        mydisp = phi[i::dof_per_node]
        # print(f"{eta.shape=} {mydisp.shape=}")
        axs[imode].plot(eta, mydisp, label=f"{disp_strs[i]}-{imode}")

    axs[imode].legend()
    # plt.legend()
plt.show()
print(f"{freqs[:6]=}")

# can use this to then write out the solution to a VTK file
# ---------------------------------------------------------