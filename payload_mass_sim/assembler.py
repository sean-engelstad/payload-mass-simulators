__all__ = ["Material", "TreeData", "BeamAssembler"]

import numpy as np
import matplotlib.pyplot as plt
import time
import niceplots
from .tree_data import TreeData
from .material import Material
from ._TS_elem import *
from scipy.linalg import eigh

class BeamAssembler:
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
        self.Fr = None

        self.freq_err_hist = []
        self.freq_hist = []

    def _build_dense_matrices(self, x):

        # get new xpts
        lengths = x[0::3]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # init matrices
        K = np.zeros((self.ndof, self.ndof))
        M = np.zeros((self.ndof, self.ndof))

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[3*icomp]
            t1 = x[3*icomp+1]
            t2 = x[3*icomp+2]

            # determine element orientation for first elem in comp group
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            # ref_axis = np.zeros((3,))
            # ref_axis[rem_orient_ind[0]] = 1.0
            ref_axis = np.array([0.0, 1.0, 0.0]) # see below we rotate from x-dir later

            # print(f"{ref_axis=}")

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            # elem_xpts = np.concatenate([xpt1, xpt2], axis=0)
            # switch to xpts in x-dir then permute x,y,z
            elem_xpts = np.array([0.0] * 3 + [L/nelem_per_comp, 0.0, 0.0])
            qvars = np.array([0.0]*12)

            # get the Kelem and Melem for this component
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]
            CM = Cfull[6:]
            Kelem = get_stiffness_matrix(elem_xpts, qvars, ref_axis, CK)
            Melem = get_mass_matrix(elem_xpts, qvars, ref_axis, CM)

            # now only need to rotate Kelem
            if orient_ind == 2:
                perm_ind0 = np.array([2, 1, 0])
            elif orient_ind == 1:
                perm_ind0 = np.array([1, 2, 0])
            elif orient_ind == 2:
                perm_ind0 = np.array([0, 2, 1])
            perm_ind = np.concatenate([perm_ind0, perm_ind0+3, perm_ind0+6, perm_ind0+9], axis=0)

            Kelem = Kelem[perm_ind,:][:,perm_ind]
            Melem = Melem[perm_ind,:][:,perm_ind] # not really necessary to permute this one

            # now do assembly step for each element
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                # print(f"{ielem=} {elem_nodes=}")
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                # print(f"{glob_dof=}")
                rows, cols = np.ix_(glob_dof, glob_dof)
                K[rows, cols] += Kelem
                M[rows,cols] += Melem

        # done with assembly procedure --------------
        # plt.imshow(K)
        # plt.show()

        # print out diagonal of stiffnesses to terminal
        # Kdiag = np.diag(K)
        # for i in range(6*10):
        #     inode = i // 6
        #     idof = i % 6
        #     val = K[i,i]
        #     dof_str = ["u","v","w","thx","thy", "thz"][idof]
        #     print(f"Kdiag @ {dof_str}{inode} = {val:.4e}")
        # exit()

        # apply bcs for reduced matrix --------------
        bcs = [_ for _ in range(6)]
        self.keep_dof = [_ for _ in range(self.ndof) if not(_ in bcs)]
        self.Kr = K[self.keep_dof,:][:,self.keep_dof]
        self.Mr = M[self.keep_dof,:][:,self.keep_dof]

        # plt.imshow(self.Kr)
        # plt.show()

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

    def get_mass(self, x):
        # get mass of entire structure
        rho = self.material.rho
        V = 0
        Varray = np.array([x[3*icomp+2]*x[3*icomp+1]*x[3*icomp] for icomp in range(self.ncomp)])
        # for icomp in range(self.ncomp):
        #     V+= (x[3*icomp+2]*x[3*icomp+1]*x[3*icomp]) # t2*t1*l
        return rho*np.sum(Varray)

    def _solve_static(self, x):
        # TODO : add in _build_inertial_loads again..
        # computes Kr * ur = Fr in reduced system
        self._build_dense_matrices(x)
        self._build_inertial_loads(x)

        self.ur = np.linalg.solve(self.Kr, self.Fr)
        self.u = np.zeros((self.ndof,))
        self.u[self.keep_dof] = self.ur[:]
        print(F"{self.u=}")

        # now plot the disps of linear static?
        return self.u

    # SENSITIVITIES --------------------------

    def get_mass_gradient(self, x):
        # compute mass gradient
        # TODO: look at individual derivatives
        dmdx_grad = np.array([0.0]*3*self.ncomp)
        for icomp in range(self.ncomp):
            dmdx_grad[3*icomp] = self.material.rho*x[3*icomp+1]*x[3*icomp+2] # dm/dl
            dmdx_grad[3*icomp+1] = self.material.rho*x[3*icomp]*x[3*icomp+2] # dm/dt1
            dmdx_grad[3*icomp+2] = self.material.rho*x[3*icomp]*x[3*icomp+1] # dm/dt2
        return dmdx_grad

    def get_frequency_gradient(self, x, imode):
        # get the DV gradient of natural frequencies for optimization
        freq = self.freqs[imode]
        num = self._get_dKdx_term(x, imode) - freq**2 * self._get_dMdx_term(x, imode)
        den = self._get_modal_mass(imode) * 2 * freq
        return num / den

    def _get_modal_mass(self, imode):
        # phi^T * M * phi modal mass for single eigenmode with already computed M matrix
        phi_red = self.eigvecs[:,imode]
        return np.dot(np.dot(self.Mr, phi_red), phi_red)

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

    def _get_dKdx_term(self, x, imode):
        # use trick to get phi^T * dK/dx * phi
        # namely find dU/dx at element level with Ue(phi,x) of ue^T * Ke * ue with ue = phi\
        dKdx_grad = np.zeros(3 * self.ncomp)

        # get relevant eigenvector
        phi_red = self.eigvecs[:,imode]
        phi = np.zeros((self.ndof,))
        phi[self.keep_dof] = phi_red[:]

        # loop over each component to get local DV derivs
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[3*icomp]
            t1 = x[3*icomp+1]
            t2 = x[3*icomp+2]

            # prelim --------------
            # get orient ind and ref axis
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            ref_axis = np.array([0.0, 1.0, 0.0]) # see below we rotate from x-dir later
            elem_xpts0 = np.array([0.0] * 3 + [L/nelem_per_comp, 0.0, 0.0])

            # apply inv perm to phi instead of on Kelem
            if orient_ind == 2:
                perm_ind0 = np.array([2, 1, 0])
            elif orient_ind == 1:
                perm_ind0 = np.array([1, 2, 0])
            elif orient_ind == 2:
                perm_ind0 = np.array([0, 2, 1])
            perm_ind = np.concatenate([perm_ind0, perm_ind0+3, perm_ind0+6, perm_ind0+9], axis=0)
            iperm = np.zeros((12,))
            for i in range(12):
                j = perm[i]
                iperm[j] = i

            # initial const data
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]; CM = Cfull[6:]

            # now do assembly step for each element -----
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                phie = phi[glob_dof]
                phie = phie[iperm] # inv permute so don't need to permute Kelem / strain energy for convenience

                detXd = L / nelem_per_comp / 2.0

                # compute dK/dL term ------------
                h = 1e-30 # complex-step method on first order
                pert_xpts = np.array([0.0] * 3 + [1/nelem_per_comp, 0.0, 0.0])
                dKdx_grad[3*icomp] += np.imag(get_strain_energy(
                    elem_xpts0 + pert_xpts * 1j * h,
                    phie, ref_axis, CK
                )) / h / (0.5 * detXd)

                # compute dK/t1 term ------------
                Cd1 = get_constitutive_data(self.material, t1 + h * 1j, t2)
                dKdx_grad[3*icomp+1] += np.imag(get_strain_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd1[:6]
                )) / h / (0.5 * detXd)

                # compute dK/t2 term ------------
                Cd2 = get_constitutive_data(self.material, t1, t2 + h * 1j)
                dKdx_grad[3*icomp+1] += np.imag(get_strain_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd2[:6]
                )) / h / (0.5 * detXd)

    def _get_dMdx_term(self, x, imode):
        # use trick to get phi^T * dM/dx * phi
        # namely find dU/dx at element level with Te(phi,x) of ue^T * Me * ue with ue = phi
        dKdx_grad = np.zeros(3 * self.ncomp)

        # get relevant eigenvector
        phi_red = self.eigvecs[:,imode]
        phi = np.zeros((self.ndof,))
        phi[self.keep_dof] = phi_red[:]

        # loop over each component to get local DV derivs
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[3*icomp]
            t1 = x[3*icomp+1]
            t2 = x[3*icomp+2]

            detXd = L / nelem_per_compj / 2.0

            # prelim --------------
            # get orient ind and ref axis
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            ref_axis = np.array([0.0, 1.0, 0.0]) # see below we rotate from x-dir later
            elem_xpts0 = np.array([0.0] * 3 + [L/nelem_per_comp, 0.0, 0.0])

            # apply inv perm to phi instead of on Kelem
            if orient_ind == 2:
                perm_ind0 = np.array([2, 1, 0])
            elif orient_ind == 1:
                perm_ind0 = np.array([1, 2, 0])
            elif orient_ind == 2:
                perm_ind0 = np.array([0, 2, 1])
            perm_ind = np.concatenate([perm_ind0, perm_ind0+3, perm_ind0+6, perm_ind0+9], axis=0)
            iperm = np.zeros((12,))
            for i in range(12):
                j = perm[i]
                iperm[j] = i

            # initial const data
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]; CM = Cfull[6:]

            # now do assembly step for each element -----
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                phie = phi[glob_dof]
                phie = phie[iperm] # inv permute so don't need to permute Kelem / strain energy for convenience

                # compute dK/dL term ------------
                h = 1e-30 # complex-step method on first order
                pert_xpts = np.array([0.0] * 3 + [1/nelem_per_comp, 0.0, 0.0])
                dKdx_grad[3*icomp] += np.imag(get_kinetic_energy(
                    elem_xpts0 + pert_xpts * 1j * h,
                    phie, ref_axis, CM
                )) / h / (0.5 * detXd)

                # compute dK/t1 term ------------
                Cd1 = get_constitutive_data(self.material, t1 + h * 1j, t2)
                dKdx_grad[3*icomp+1] += np.imag(get_kinetic_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd1[6:]
                )) / h / (0.5 * detXd)

                # compute dK/t2 term ------------
                Cd2 = get_constitutive_data(self.material, t1, t2 + h * 1j)
                dKdx_grad[3*icomp+1] += np.imag(get_kinetic_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd2[6:]
                )) / h / (0.5 * detXd)

    # PLOT UTILS -------------------------

    def _plot_xpts(self, new_xpts, color):
        for ielem in range(self.nelems):
            nodes = self.elem_conn[ielem]
            xpt1 = new_xpts[3*nodes[0]:3*nodes[0]+3]
            xpt2 = new_xpts[3*nodes[1]:3*nodes[1]+3]
            xv = [xpt1[0], xpt2[0]]
            yv = [xpt1[1], xpt2[1]]
            zv = [xpt1[2], xpt2[2]]
            plt.plot(xv, yv, zv, 'o', color=color, linewidth=2)
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