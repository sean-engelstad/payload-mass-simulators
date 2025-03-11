# now let's test this out and visualize it
import sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
sys.path.append("beam-fea")
from beam_elem import *
from scipy.linalg import eigh
import os

if not os.path.exists("_modal"):
    os.mkdir("_modal")

# axial torsion, and transverse beam oriented in x direction
# so y,z are transverse directions

# assume each node has 6 DOF (although of course in this case)
# here are the material properties

rho = 2.7e3 # kg/m^3
E = 70e9 # aluminum (Pa)
nu = 0.3
G = E / 2.0 / (1 + nu)

# mesh connectivity and nodes
# -------------------------------------
# now we're doing the first level of tuning fork, base + 4 lateral + 4 vertical
# 11 nodes per component

# design parameters
base_h = 0.3
x_lat = 0.05
y_lat = 0.05
x_h = 0.5
y_h = 0.5

# lists for FEA data
xpts = [0.0,0.0,0.0]
elem_conn = []
nxe_per_comp = 10 # shorthand npc
npc = nxe_per_comp
elem_comp = []

# base rod in +z direction
# nodes 0 through npc inclusive
icomp = 0
for ielem in range(nxe_per_comp):
    elem_conn += [[ielem, ielem+1]]
    zfrac = 1.0 * (ielem+1) / nxe_per_comp
    xpts += [0.0, 0.0, base_h * zfrac]
    # xpts += [base_h * zfrac, 0.0, 0.0]
    elem_comp += [icomp]
# ends with node npc

# four lateral rods
# xp rod
icomp = 1
for ielem in range(npc):
    elem_conn += [[icomp*npc+ielem, icomp*npc+ielem+1]]
    frac = 1.0 * (ielem+1) / npc
    xpts += [frac * x_lat, 0.0, base_h]
    elem_comp += [icomp]
# npc+1 to 2*npc

# xn rod
icomp = 2
for ielem in range(npc):
    if ielem == 0:
        elem_conn += [[npc, icomp*npc+1]]
    else:
        elem_conn += [[icomp*npc+ielem, icomp*npc+ielem+1]]
    frac = 1.0 * (ielem+1) / npc
    xpts += [-frac * x_lat, 0.0, base_h]
    elem_comp += [icomp]
# 2*npc+1 to 3*npc

# yp rod
icomp = 3
for ielem in range(npc):
    if ielem == 0:
        elem_conn += [[npc, icomp*npc+1]]
    else:
        elem_conn += [[icomp*npc+ielem, icomp*npc+ielem+1]]
    frac = 1.0 * (ielem+1) / npc
    xpts += [0.0, frac * y_lat, base_h]
    elem_comp += [icomp]
# 3*npc+1 to 4*npc

# yn rod
icomp = 4
for ielem in range(npc):
    if ielem == 0:
        elem_conn += [[npc, icomp*npc+1]]
    else:
        elem_conn += [[icomp*npc+ielem, icomp*npc+ielem+1]]
    frac = 1.0 * (ielem+1) / npc
    xpts += [0.0, -frac * y_lat, base_h]
    elem_comp += [icomp]
# 4*npc+1 to 5*npc

# vert rods
# xp vert rod
icomp = 5
for ielem in range(npc):
    if ielem == 0:
        elem_conn += [[2*npc, icomp*npc+1]]
    else:
        elem_conn += [[icomp*npc+ielem, icomp*npc+ielem+1]]
    frac = 1.0 * (ielem+1) / npc
    xpts += [x_lat, 0.0, base_h + frac * x_h]
    elem_comp += [icomp]

# xn vert rod
icomp = 6
for ielem in range(npc):
    if ielem == 0:
        elem_conn += [[3*npc, icomp*npc+1]]
    else:
        elem_conn += [[icomp*npc+ielem, icomp*npc+ielem+1]]
    frac = 1.0 * (ielem+1) / npc
    xpts += [-x_lat, 0.0, base_h + frac * x_h]
    elem_comp += [icomp]

# yp vert rod
icomp = 7
for ielem in range(npc):
    if ielem == 0:
        elem_conn += [[4*npc, icomp*npc+1]]
    else:
        elem_conn += [[icomp*npc+ielem, icomp*npc+ielem+1]]
    frac = 1.0 * (ielem+1) / npc
    xpts += [0.0, y_lat, base_h + frac * y_h]
    elem_comp += [icomp]

# yn vert rod
icomp = 8
for ielem in range(npc):
    if ielem == 0:
        elem_conn += [[5*npc, icomp*npc+1]]
    else:
        elem_conn += [[icomp*npc+ielem, icomp*npc+ielem+1]]
    frac = 1.0 * (ielem+1) / npc
    xpts += [0.0, -y_lat, base_h + frac * y_h]
    elem_comp += [icomp]

# now when done convert to np.ndarrays
xpts = np.array(xpts)

# plot the mesh connectivity to check
plot_conn = True
if plot_conn:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for ielem in range(len(elem_conn)):
        nodes = elem_conn[ielem]
        xpt1 = xpts[3*nodes[0]:3*nodes[0]+3]
        xpt2 = xpts[3*nodes[1]:3*nodes[1]+3]
        xv = [xpt1[0], xpt2[0]]
        yv = [xpt1[1], xpt2[1]]
        zv = [xpt1[2], xpt2[2]]
        plt.plot(xv, yv, zv, linewidth=2)
    # plt.show()
    plt.savefig("_modal/mesh.png", dpi=400)
    # exit()

# get some data from mesh conn (counts)
ncomp = icomp + 1
nnodes = xpts.shape[0] // 3
nelems = len(elem_conn)

thick_vars = [0.01] * 2 * ncomp

print(f"Mesh with {nnodes=} {nelems=} {ncomp=}")

# construct the global stiffness matrix
# -------------------------------------
dof_per_node = 6
ndof = nnodes * dof_per_node
K = np.zeros((ndof, ndof))
M = np.zeros((ndof, ndof))

# dof per node are [u,v,w,thx,thy,thz]

for ielem in range(nelems):

    # determine the orientation of the element (x, y, or z oriented)
    nodes = elem_conn[ielem]
    node1 = nodes[0]; node2 = nodes[1]
    xpt1 = xpts[3*node1:3*node1+3]; xpt2 = xpts[3*node2:3*node2+3]
    dxpt = xpt2 - xpt1
    L_elem = np.linalg.norm(dxpt)
    # print(f"{L_elem=}")
    orient_list = ['x', 'y', 'z']
    orient_ind = np.argmax(np.abs(dxpt))
    orient = orient_list[orient_ind]
    # print(f"{orient=}")
    # exit()

    rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
    # print(f"{rem_orient_ind=}")
    # exit()
    # print(f"{ielem=} {orient_ind=} {rem_orient_ind=}")

    # get the cross section dimensions (from thickness DVs)
    icomp = elem_comp[ielem]
    t1 = thick_vars[2*icomp]
    t2 = thick_vars[2*icomp+1]
    A = t1 * t2
    I1 = t2**3 * t1 / 12.0
    I2 = t1**3 * t2 / 12.0
    J = I1 + I2
    # print(f"{ielem=} {I1=} {I2=} {J=}")

    # get element stiffness matrices
    # xscale = L_elem # length of beam
    K_ax = get_kelem_axial(E, A, L_elem)
    K_tor = get_kelem_torsion(G, J, L_elem)
    K1_tr = get_kelem_transverse(L_elem, E, I1)
    K2_tr = get_kelem_transverse(L_elem, E, I2)

    # get element mass matrices
    M_ax = get_melem_axial(rho, A, L_elem)
    M_tor = get_melem_torsion(rho, A, J, L_elem)
    M1_tr = get_melem_transverse(L_elem, rho, A)
    M2_tr = get_melem_transverse(L_elem, rho, A)

    # figure out where to add this in global matrix
    axial_nodal_dof = [orient_ind]
    torsion_nodal_dof = [3+orient_ind]
    ind1 = rem_orient_ind[0]
    ind2 = rem_orient_ind[1]
    tr1_nodal_dof = [ind1, 3+ind2]
    tr2_nodal_dof = [ind2, 3+ind1]

    # print(f"{ielem}: {axial_nodal_dof=} {torsion_nodal_dof=} {tr1_nodal_dof=} {tr2_nodal_dof=}")

    # get global dof for each physics and this element
    axial_dof = [6*inode + _dof for _dof in axial_nodal_dof for inode in nodes]
    torsion_dof = [6*inode + _dof for _dof in torsion_nodal_dof for inode in nodes]
    tr1_dof = np.sort([6*inode + _dof for _dof in tr1_nodal_dof for inode in nodes])
    tr2_dof = np.sort([6*inode + _dof for _dof in tr2_nodal_dof for inode in nodes])

    # print(f"\t{axial_dof=} {torsion_dof=} {tr1_dof=} {tr2_dof=}")

    # get assembly arrays
    ax_rows, ax_cols = np.ix_(axial_dof, axial_dof)
    tor_rows, tor_cols = np.ix_(torsion_dof, torsion_dof) # thx spots
    tr1_rows, tr1_cols = np.ix_(tr1_dof, tr1_dof)
    tr2_rows, tr2_cols = np.ix_(tr2_dof, tr2_dof)

    # print(f'{axial_dof=} {torsion_dof=} {ytr_dof=} {ztr_dof=}')
    # exit()

    # assemble to global stiffness matrix
    K[ax_rows, ax_cols] += K_ax
    K[tor_rows, tor_cols] += K_tor
    K[tr1_rows, tr1_cols] += K1_tr
    K[tr2_rows, tr2_cols] += K2_tr

    # assemble to global mass matrix
    M[ax_rows, ax_cols] += M_ax
    M[tor_rows, tor_cols] += M_tor
    M[tr1_rows, tr1_cols] += M1_tr
    M[tr2_rows, tr2_cols] += M2_tr

    # TODO: want to make it sparse matrix later

# exit()

# apply bcs to get reduced matrix
# ------------------------------------------

# fix u,v,w,thx,thy,thz at 0
bcs = [_ for _ in range(6)]

# no we actually need to get reduced stiffness and mass matrices
# otherwise non-zero eigen displacements can occur at BCs which doesn't make sense
keep_dof = [_ for _ in range(ndof) if not(_ in bcs)]
Kr = K[keep_dof,:][:,keep_dof]
Mr = M[keep_dof,:][:,keep_dof]

plot_Kr = True
if plot_Kr:
    plt.close('all')
    norm = colors.SymLogNorm(vmin=-10, vmax=10, linthresh=1e-4)
    plt.imshow(Kr, norm=norm)
    plt.colorbar()
    plt.savefig("_modal/Kr.png")
    plt.close('all')
    # plt.show()
    print(f"{Kr=}")

# solve eigenvalue problem
# ------------------------

# solve symmetric eigenvalue problem
eigvals, eigvecs = eigh(Kr, Mr)
freqs = np.sqrt(eigvals)
print(f"{freqs[:6]=}")
# plt.imshow(eigvecs[:,:5])
# plt.show()

# plot undeformed and deformed shapes in 3D
import os
if not os.path.exists("_modal"):
    os.mkdir("_modal")

def plot_xpts(new_xpts, color):
    for ielem in range(nelems):
        nodes = elem_conn[ielem]
        xpt1 = new_xpts[3*nodes[0]:3*nodes[0]+3]
        xpt2 = new_xpts[3*nodes[1]:3*nodes[1]+3]
        xv = [xpt1[0], xpt2[0]]
        yv = [xpt1[1], xpt2[1]]
        zv = [xpt1[2], xpt2[2]]
        plt.plot(xv, yv, zv, color=color, linewidth=2)
    return

show = False
for imode in range(4):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    phi_r = eigvecs[:,imode]
    phi = np.zeros((ndof,))
    phi[keep_dof] = phi_r
    uvw_ind = np.array(np.sort([6*inode+dof for dof in range(3) for inode in range(nnodes)]))
    # print(f"{uvw_ind=}")
    uvw_phi = phi[uvw_ind]

    # normalize phi
    uvw_norm = np.linalg.norm(uvw_phi)
    # print(f"{uvw_norm=}")
    phi_p = uvw_phi / uvw_norm
    # print(f"{phi_p=}")
    # exit()
    # choose scale factor
    scale = 0.1
    phi_p *= scale

    # print(f'{phi_p=}')
    # print(f"{phi_p.shape=} {xpts.shape=}")
    # exit()

    # plot undeformed
    plot_xpts(xpts, 'k')
    plot_xpts(xpts + phi_p, 'b')
    # ax.set_layout('tight')
    # plt.tight_layout()
    # ax.set_box_aspect([1,1,1])
    if show:
        plt.show()
    else:
        plt.savefig(f"_modal/mode{imode}.png")

# plt.legend()
# plt.show()

# can use this to then write out the solution to a VTK file
# ---------------------------------------------------------