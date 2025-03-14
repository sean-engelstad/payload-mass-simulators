__all__ = ["Material", "TreeData", "Beam3DTree"]

import numpy as np
import matplotlib.pyplot as plt
import time
import niceplots
from tree_data import TreeData
from material import Material
from beam_elem import *
from scipy.linalg import eigh


class Beam3DTree:
    def __init__(self, material:Material, tree:TreeData):
        self.material = material
        self.tree = tree

        # some important metadata quantities
        self.nnodes = self.tree.nnodes
        self.nelems = self.tree.nelems
        self.ncomp = self.tree.ncomp
        self.elem_conn = self.tree.elem_conn
        self.elem_comp = self.tree.elem_comp
        self.dof_per_node = 6
        self.ndof = self.dof_per_node * self.nnodes
        self.red_ndof = self.ndof - self.dof_per_node # eliminates root node with cantilever
        self.nelem_per_comp = self.tree.nelem_per_comp
        self.num_dvs = self.ncomp * 3

        # get material data into this class
        self.E = self.material.E
        self.nu = self.material.nu
        self.G = self.material.G
        self.rho = self.material.rho

        self.xpts = None
        self.Kr = None
        self.Mr = None
        self.keep_dof = None # for bc removal
        self.freqs = None
        self.eigvecs = None

        self.freq_err_hist = []
        self.freq_hist = []

    def _build_sparse_matrices(self, x):
        # TODO : use design variables to compute K, M as sparse BCSR matrices
        pass

    def _build_dense_matrices(self, x):
        # x is a vector of DVs [l1, t1, t2]*nnodes for each node repeating

        # get new xpts
        lengths = x[0::3]
        self.xpts = self.tree.get_xpts(lengths)

        K = np.zeros((self.ndof, self.ndof))
        M = np.zeros((self.ndof, self.ndof))

        for ielem in range(self.nelems):

            # determine element orientation
            nodes = self.elem_conn[ielem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])

            # get local design variables
            icomp = self.elem_comp[ielem]
            length = x[3*icomp]
            t1 = x[3*icomp+1]
            t2 = x[3*icomp+2]

            # get cross section dimensions from thickness DVs
            L_elem = length / self.nelem_per_comp
            A = t1 * t2
            I1 = t2**3 * t1 / 12.0
            I2 = t1**3 * t2 / 12.0
            J = I1 + I2
            
            # get element stiffness matrices
            K_ax = self.E * A / L_elem * get_kelem_axial()
            K_tor = self.G * J / L_elem * get_kelem_torsion()
            K1_tr = self.E * I1 / L_elem**3 * get_kelem_transverse()
            K2_tr = self.E * I2 / L_elem**3 * get_kelem_transverse()

            # get element mass matrices
            M_ax = self.rho * A * L_elem / 6 * get_melem_axial()
            M_tor = self.rho * A * J * L_elem / 3 * get_melem_torsion()
            M1_tr = self.rho * A * L_elem * get_melem_transverse()
            M2_tr = self.rho * A * L_elem * get_melem_transverse()

            # figure out which element nodes correspond to axial, torsion, transverse depending
            # on the beam orientation
            axial_nodal_dof = [orient_ind]
            torsion_nodal_dof = [3+orient_ind]
            ind1 = rem_orient_ind[0]; ind2 = rem_orient_ind[1]
            tr1_nodal_dof = [ind1, 3+ind2]; tr2_nodal_dof = [ind2, 3+ind1]

            # get global dof for each physics and this element
            axial_dof = [6*inode + _dof for _dof in axial_nodal_dof for inode in nodes]
            torsion_dof = [6*inode + _dof for _dof in torsion_nodal_dof for inode in nodes]
            tr1_dof = np.sort([6*inode + _dof for _dof in tr1_nodal_dof for inode in nodes])
            tr2_dof = np.sort([6*inode + _dof for _dof in tr2_nodal_dof for inode in nodes])

            # get assembly arrays
            ax_rows, ax_cols = np.ix_(axial_dof, axial_dof)
            tor_rows, tor_cols = np.ix_(torsion_dof, torsion_dof) # thx spots
            tr1_rows, tr1_cols = np.ix_(tr1_dof, tr1_dof)
            tr2_rows, tr2_cols = np.ix_(tr2_dof, tr2_dof)

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

        # apply reduced bcs to the matrix
        bcs = [_ for _ in range(6)] # just the first node of root fixed all DOF
        self.keep_dof = [_ for _ in range(self.ndof) if not(_ in bcs)]
        self.Kr = K[self.keep_dof,:][:,self.keep_dof]
        self.Mr = M[self.keep_dof,:][:,self.keep_dof]

    def get_frequencies(self, x, nmodes=5):

        # update xpts and assemble Kr, Mr matrices
        # self._build_sparse_matrices(x) # TODO
        self._build_dense_matrices(x)

        # solve the generalized eigenvalues problem
        print("Solving eigenvalue problem:")
        start_time = time.time()
        eigvals, self.eigvecs = eigh(self.Kr, self.Mr)
        self.freqs = np.sqrt(eigvals)
        dt = time.time() - start_time
        print(f"\tsolved eigvals in {dt:.4f} seconds.")

        return self.freqs[:nmodes]
    
    def get_frequency_gradient(self, x, imode):
        # get the DV gradient of natural frequencies for optimization
        freq = self.freqs[imode]
        num = self._get_dKdx_term(x, imode) - freq**2 * self._get_dMdx_term(x, imode)
        den = self._get_modal_mass(imode) * 2 * freq
        return num / den

    def get_mass(self, x):
        # get mass of entire structure
        rho = self.material.rho
        V = 0
        Varray = np.array([x[3*icomp+2]*x[3*icomp+1]*x[3*icomp] for icomp in range(self.ncomp)])
        # for icomp in range(self.ncomp):
        #     V+= (x[3*icomp+2]*x[3*icomp+1]*x[3*icomp]) # t2*t1*l
        return rho*np.sum(Varray)

    def get_mass_gradient(self, x):
        # compute mass gradient
        # TODO: look at individual derivatives
        dmdx_grad = np.array([0.0]*3*self.ncomp)
        for icomp in range(self.ncomp):
            dmdx_grad[3*icomp] = self.material.rho*x[3*icomp+1]*x[3*icomp+2] # dm/dl
            dmdx_grad[3*icomp+1] = self.material.rho*x[3*icomp]*x[3*icomp+2] # dm/dt1
            dmdx_grad[3*icomp+2] = self.material.rho*x[3*icomp]*x[3*icomp+1] # dm/dt2
        return dmdx_grad
    
        # figure out inline later?
        # dmdx_grad = np.array([self.material.rho*x[3*icomp+1]*x[3*icomp+2], self.material.rho*x[3*icomp]*x[3*icomp+2], self.material.rho*x[3*icomp]*x[3*icomp+1] for icomp in range(self.ncomp)])

    def _get_dKdx_term(self, x, imode):
        # get phi^T * dK/dx * phi term for single eigenmode, gradient of all DVs
        dKdx_grad = np.array([0.0]*3*self.ncomp)

        # get the eigenvector (in full coordinates, zeroed out at root)
        # so no need to check BCs in this loop
        phi_red = self.eigvecs[:,imode]
        phi = np.zeros((self.ndof,))
        phi[self.keep_dof] = phi_red[:]

        for ielem in range(self.nelems):
            
            # determine element orientation
            nodes = self.elem_conn[ielem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])

            # get local design variables
            icomp = self.elem_comp[ielem]
            length = x[3*icomp]
            t1 = x[3*icomp+1]
            t2 = x[3*icomp+2]

            # figure out which element nodes correspond to axial, torsion, transverse depending
            # on the beam orientation
            axial_nodal_dof = [orient_ind]
            torsion_nodal_dof = [3+orient_ind]
            ind1 = rem_orient_ind[0]; ind2 = rem_orient_ind[1]
            tr1_nodal_dof = [ind1, 3+ind2]; tr2_nodal_dof = [ind2, 3+ind1]

            # get global dof for each physics and this element
            axial_dof = [6*inode + _dof for _dof in axial_nodal_dof for inode in nodes]
            torsion_dof = [6*inode + _dof for _dof in torsion_nodal_dof for inode in nodes]
            tr1_dof = np.sort([6*inode + _dof for _dof in tr1_nodal_dof for inode in nodes])
            tr2_dof = np.sort([6*inode + _dof for _dof in tr2_nodal_dof for inode in nodes])

            # now compute the element components of the eigenvector for each physics term
            phie_ax = phi[axial_dof]
            phie_tor = phi[torsion_dof]
            phie_tr1 = phi[tr1_dof]
            phie_tr2 = phi[tr2_dof] # each physics is decoupled
            # that is we're going to break this down into element quadratic products
            # phi_e^T K_e phi_e, except also for each decoupled physics K_e as well

            # cross section dimensions
            L_elem = length / self.nelem_per_comp
            A = t1 * t2
            I1 = t2**3 * t1 / 12.0
            I2 = t1**3 * t2 / 12.0
            J = I1 + I2

            # local elem grad
            local_grad = [0]*3

            def get_phys_grads(dK_ax, dK_tor, dK1_tr, dK2_tr):
                # for single DV
                deriv = 0.0
                deriv += np.dot(np.dot(dK_ax, phie_ax), phie_ax)
                deriv += np.dot(np.dot(dK_tor, phie_tor), phie_tor)
                deriv += np.dot(np.dot(dK1_tr, phie_tr1), phie_tr1)
                deriv += np.dot(np.dot(dK2_tr, phie_tr2), phie_tr2)
                return deriv

            # now compute stiffness matrix derivatives
            # ----------------------------------------

            # for reference, A, I1, I2 written explicitly in terms of DVs
            K_ax = self.E * A / L_elem * get_kelem_axial()
            K_tor = self.G * J / L_elem * get_kelem_torsion()
            K1_tr = self.E * I1 / L_elem**3 * get_kelem_transverse()
            K2_tr = self.E * I2 / L_elem**3 * get_kelem_transverse()

            # first the length derivatives
            dL_elem = 1.0 / self.nelem_per_comp
            local_grad[0] = dL_elem * get_phys_grads(
                dK_ax=K_ax * -1.0 / L_elem,
                dK_tor=K_tor * -1.0 / L_elem,
                dK1_tr=K1_tr * -3.0 / L_elem,
                dK2_tr=K2_tr * -3.0 / L_elem
            )

            # second the t1 thickness derivatives
            local_grad[1] = get_phys_grads(
                dK_ax=K_ax / t1,
                dK_tor=K_tor / J * (I1 * 1.0 + I2 * 3.0) / t1,
                dK1_tr=K1_tr * 1.0 / t1,
                dK2_tr=K2_tr * 3.0 / t1,
            )

            # second the t2 thickness derivatives
            local_grad[2] = get_phys_grads(
                dK_ax=K_ax / t2,
                dK_tor=K_tor / J * (I1 * 3.0 + I2 * 1.0) / t2,
                dK1_tr=K1_tr * 3.0 / t2,
                dK2_tr=K2_tr * 1.0 / t2,
            )

            # ---------------------

            # add back local grad to global grad
            dKdx_grad[3*icomp:3*icomp+3] += np.array(local_grad)

        return dKdx_grad

    def _get_dMdx_term(self, x, imode):
        # phi^T * dM/dx * phi for single eigenmode, gradient among all DVs
        dMdx_grad = np.array([0.0]*3*self.ncomp)

        # get the eigenvector (in full coordinates, zeroed out at root)
        # so no need to check BCs in this loop
        phi_red = self.eigvecs[:,imode]
        phi = np.zeros((self.ndof,))
        phi[self.keep_dof] = phi_red[:]

        for ielem in range(self.nelems):
            
            # determine element orientation
            nodes = self.elem_conn[ielem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])

            # get local design variables
            icomp = self.elem_comp[ielem]
            length = x[3*icomp]
            t1 = x[3*icomp+1]
            t2 = x[3*icomp+2]

            # figure out which element nodes correspond to axial, torsion, transverse depending
            # on the beam orientation
            axial_nodal_dof = [orient_ind]
            torsion_nodal_dof = [3+orient_ind]
            ind1 = rem_orient_ind[0]; ind2 = rem_orient_ind[1]
            tr1_nodal_dof = [ind1, 3+ind2]; tr2_nodal_dof = [ind2, 3+ind1]

            # get global dof for each physics and this element
            axial_dof = [6*inode + _dof for _dof in axial_nodal_dof for inode in nodes]
            torsion_dof = [6*inode + _dof for _dof in torsion_nodal_dof for inode in nodes]
            tr1_dof = np.sort([6*inode + _dof for _dof in tr1_nodal_dof for inode in nodes])
            tr2_dof = np.sort([6*inode + _dof for _dof in tr2_nodal_dof for inode in nodes])

            # now compute the element components of the eigenvector for each physics term
            phie_ax = phi[axial_dof]
            phie_tor = phi[torsion_dof]
            phie_tr1 = phi[tr1_dof]
            phie_tr2 = phi[tr2_dof] # each physics is decoupled
            # that is we're going to break this down into element quadratic products
            # phi_e^T K_e phi_e, except also for each decoupled physics K_e as well

            # cross section dimensions
            L_elem = length / self.nelem_per_comp
            A = t1 * t2
            I1 = t2**3 * t1 / 12.0
            I2 = t1**3 * t2 / 12.0
            J = I1 + I2

            # local elem grad
            local_grad = [0]*3

            def get_phys_grads(dM_ax, dM_tor, dM1_tr, dM2_tr):
                # for single DV
                deriv = 0.0
                deriv += np.dot(np.dot(dM_ax, phie_ax), phie_ax)
                deriv += np.dot(np.dot(dM_tor, phie_tor), phie_tor)
                deriv += np.dot(np.dot(dM1_tr, phie_tr1), phie_tr1)
                deriv += np.dot(np.dot(dM2_tr, phie_tr2), phie_tr2)
                return deriv

            # now compute stiffness matrix derivatives
            # ----------------------------------------

            # for reference, get element mass matrices
            M_ax = self.rho * A * L_elem / 6 * get_melem_axial()
            M_tor = self.rho * A * J * L_elem / 3 * get_melem_torsion()
            M1_tr = self.rho * A * L_elem * get_melem_transverse()
            M2_tr = self.rho * A * L_elem * get_melem_transverse()

            # first the length derivatives
            dL_elem = 1.0 / self.nelem_per_comp
            local_grad[0] = dL_elem * get_phys_grads(
                dM_ax=M_ax / L_elem,
                dM_tor=M_tor / L_elem,
                dM1_tr=M1_tr / L_elem,
                dM2_tr=M2_tr / L_elem
            )

            # second the t1 thickness derivatives
            local_grad[1] = get_phys_grads(
                dM_ax=M_ax / t1,
                dM_tor=M_tor / J * (I1 * 1.0 + I2 * 3.0) / t1,
                dM1_tr=M1_tr / t1,
                dM2_tr=M2_tr / t1,
            )

            # second the t2 thickness derivatives
            local_grad[2] = get_phys_grads(
                dM_ax=M_ax / t2,
                dM_tor=M_tor / J * (I1 * 3.0 + I2 * 1.0) / t2,
                dM1_tr=M1_tr / t2,
                dM2_tr=M2_tr / t2,
            )

            # ---------------------

            # add back local grad to global grad
            dMdx_grad[3*icomp:3*icomp+3] += np.array(local_grad)

        return dMdx_grad

    def _get_modal_mass(self, imode):
        # phi^T * M * phi modal mass for single eigenmode with already computed M matrix
        phi_red = self.eigvecs[:,imode]
        return np.dot(np.dot(self.Mr, phi_red), phi_red)

    def _plot_xpts(self, new_xpts, color):
        for ielem in range(self.nelems):
            nodes = self.elem_conn[ielem]
            xpt1 = new_xpts[3*nodes[0]:3*nodes[0]+3]
            xpt2 = new_xpts[3*nodes[1]:3*nodes[1]+3]
            xv = [xpt1[0], xpt2[0]]
            yv = [xpt1[1], xpt2[1]]
            zv = [xpt1[2], xpt2[2]]
            plt.plot(xv, yv, zv, color=color, linewidth=2)
        return
    
    def freq_FD_test(self, x, imode, h=1e-3):
        p_vec = np.random.rand(self.num_dvs)
        freqs = self.get_frequencies(x)
        freq_grad = self.get_frequency_gradient(x, imode)
        freqs2 = self.get_frequencies(x + p_vec * h)
        FD_val = (freqs2[imode] - freqs[imode]) / h
        HC_val = np.dot(freq_grad, p_vec)
        print(f"freq[{imode}] FD test: {FD_val=} {HC_val=}")
        return
    
    def mass_FD_test(self, x, h=1e-3):
        p_vec = np.random.rand(self.num_dvs)
        mass_0 = self.get_mass(x)
        mass_grad = self.get_mass_gradient(x)
        mass_1 = self.get_mass(x + p_vec * h)
        FD_val = (mass_1 - mass_0) / h
        HC_val = np.dot(mass_grad, p_vec)
        print(f"mass FD test: {FD_val=} {HC_val=}")
        return
    
    def dKdx_FD_test(self, x, imode, h=1e-5):
        p_vec = np.random.rand(self.num_dvs)
        self.get_frequencies(x)
        Kr0 = self.Kr.copy()
        phi = self.eigvecs[:,imode].copy()
        dKrdx_grad = self._get_dKdx_term(x, imode)
        self.get_frequencies(x + p_vec * h)
        Kr2 = self.Kr.copy()
        
        dKr_dx_p = (Kr2 - Kr0) / h
        FD_val = np.dot(np.dot(dKr_dx_p, phi), phi)
        HC_val = np.dot(p_vec, dKrdx_grad)
        print(f"dK/dx FD test: {FD_val=} {HC_val=}")
        return
    
    def dMdx_FD_test(self, x, imode, h=1e-5):
        p_vec = np.random.rand(self.num_dvs)
        self.get_frequencies(x)
        Mr0 = self.Mr.copy()
        phi = self.eigvecs[:,imode].copy()
        dMrdx_grad = self._get_dMdx_term(x, imode)
        self.get_frequencies(x + p_vec * h)
        Mr2 = self.Mr.copy()
        
        dMr_dx_p = (Mr2 - Mr0) / h
        FD_val = np.dot(np.dot(dMr_dx_p, phi), phi)
        HC_val = np.dot(p_vec, dMrdx_grad)
        print(f"dM/dx FD test: {FD_val=} {HC_val=}")
        return

    def plot_eigenmodes(self, nmodes=5, show=False, def_scale=0.3, file_prefix=""):
        for imode in range(nmodes):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            phi_r = self.eigvecs[:,imode]
            phi = np.zeros((self.ndof,))
            phi[self.keep_dof] = phi_r
            uvw_ind = np.array(np.sort([6*inode+dof for dof in range(3) for inode in range(self.nnodes)]))
            uvw_phi = phi[uvw_ind]
            uvw_phi /= np.linalg.norm(uvw_phi)

            # plot undeformed and deformed
            self._plot_xpts(self.xpts, 'k')
            self._plot_xpts(self.xpts + def_scale * uvw_phi, 'b')
            if show:
                plt.show()
            else:
                plt.savefig(f"_modal/{file_prefix}_mode{imode}.png")