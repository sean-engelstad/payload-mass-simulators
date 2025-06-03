__all__ = ["Material", "TreeData", "BeamAssemblerAdvanced"]

import numpy as np
import matplotlib.pyplot as plt
import time
import niceplots
from .tree_data import TreeData
from .material import Material
from ._TS_elem import *
from scipy.linalg import eigh
from .write_vtk import *
from .inertial_data import *

class BeamAssemblerAdvanced:
    """the advanced beam assembler has tapered thicknesses and lumped masses which the regular one does not"""
    def __init__(self, material:Material, tree:TreeData, inertial_data:InertialData=None, rho_KS:float=10.0, safety_factor:float=1.5, sparse:bool=False):
        self.material = material
        self.tree = tree
        self.inertial_data = inertial_data
        self.rho_KS = rho_KS
        self.safety_factor = safety_factor
        self.sparse = sparse

        # register some important info to the tree object
        self.tree.ndvs_per_comp = 7
        self.tree.rho = self.material.rho

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
        self.num_dvs = self.ncomp * 7 # [L, t1i, t1f, t2i, t2f, M, mx]

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
        self.ur = None
        self.u = None
        self.stresses = None
        self.thicknesses = None

        self.freq_err_hist = []
        self.freq_hist = []

        # any prelim
        # ----------
        # only need to do this one time
        if self.sparse:
            self._compute_nz_pattern()

    def _build_dense_matrices(self, x):
        """sparse matrices available, but dense eigval solver more robust atm"""

        # get new xpts
        lengths = x[0::7]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # init matrices
        K = np.zeros((self.ndof, self.ndof))
        M = np.zeros((self.ndof, self.ndof))

        tot_dt = 0.0

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            # L = x[7*icomp]
            t1i = x[7*icomp+1] # initial and final tapered thicknesses, each direction
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5] # lumped mass mag
            mx = x[7*icomp+6] # lumped mass 0 to 1 position

            # determine element orientation for first elem in comp group
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)
            # switch to xpts in x-dir then permute x,y,z
            # elem_xpts = np.array([0.0] * 3 + [L/nelem_per_comp, 0.0, 0.0])
            qvars = np.array([0.0]*12)

            # now do assembly step for each element
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                if t1 < 0 or t2 < 0:
                    print(f"{t1=} {t2=} {xi=}")

                # print(f"{t1=} {t2=}")

                # get the Kelem and Melem for this component
                Cfull = get_constitutive_data(self.material, t1, t2)
                CK = Cfull[:6]
                CM = Cfull[6:]
                # print(f"{elem_xpts=}")
                Kelem = get_stiffness_matrix(elem_xpts, qvars, ref_axis, CK)
                Melem = get_mass_matrix(elem_xpts, qvars, ref_axis, CM)

                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                # print(f"{glob_dof=}")
                rows, cols = np.ix_(glob_dof, glob_dof)
                K[rows, cols] += Kelem
                M[rows,cols] += Melem

            # add lumped mass.. in element which now contains the beam
            ielem = int(start + np.floor(mx * nelem_per_comp))
            elem_nodes = self.elem_conn[ielem]
            glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
            rows, cols = np.ix_(glob_dof, glob_dof)
            xi_start = mx % (1.0/nelem_per_comp)
            xi_elem = xi_start * nelem_per_comp
            Mi = Mmass * (1-xi_elem); Mf = Mmass * xi_elem
            Melem_lumped = np.diag([Mi]*6 + [Mf]*6) / 2.0
            # print(f'{np.diag(Melem_lumped)=}')
            M[rows,cols] += Melem_lumped

        # done with assembly procedure --------------
        # plt.imshow(K)
        # plt.show()

        # apply bcs for reduced matrix --------------
        bcs = [_ for _ in range(6)]
        self.keep_dof = [_ for _ in range(self.ndof) if not(_ in bcs)]
        self.Kr = K[self.keep_dof,:][:,self.keep_dof]
        self.Mr = M[self.keep_dof,:][:,self.keep_dof]

        # print(f"{tot_dt=}")

        # plt.imshow(self.Kr)
        # plt.show()

    def _build_inertial_loads(self, x):
        # TODO : compute inertial loads n * rho * g * A on different parts of the structure..
        assert(self.inertial_data)

        # get new xpts (kind of redundant call here, just here for deriv testing mostly)
        lengths = x[0::7]
        self.xpts = self.tree.get_xpts(lengths)

        # assuming constant load factor for now, not differentiated
        mass = self.get_mass(x)     
        nelem_per_comp = self.tree.nelem_per_comp
        # n = self.inertial_data.get_load_factor(mass)
        inertial_direc = self.inertial_data.inertial_direction
        rho = self.material.rho
        g = self.inertial_data.accel_grav

        # assemble global forces
        F = np.zeros((self.ndof,)) # global force vector

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5]
            mx = x[7*icomp+6]

            # compute element length Le
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            Le = np.linalg.norm(xpt2 - xpt1)

            # now do assembly step for each element
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                # get the xi [0,1] at start and end of element
                xi1 = ielem/nelem_per_comp
                xi2 = xi1 + 1.0 / nelem_per_comp
                t11 = t1i * (1-xi1) + t1f * xi1
                t12 = t1i * (1-xi2) + t1f * xi2
                t21 = t2i * (1-xi1) + t2f * xi1
                t22 = t2i * (1-xi2) + t2f * xi2
                S1 = t11 * t21 # CS area at start of element
                S2 = t12 * t22 # CS area at end of element
                # volume of frustum (tapered beam section)
                V = Le / 3.0 * (S1 + S2 + 0.5 * (t11 * t22 + t12 * t21))

                nodal_load_mag = rho * g * V
                # print(f"{nodal_load_mag=}")

                # don't actually need beam direc, just need to compute comp x,y,z parts of distributed load
                # compute element nodal loads
                Felem = np.zeros((12,))
                for i in range(3):
                    # this is not perfect, could distribute mass better, but it's fine for now probably
                    # just 0.5 to each node
                    Felem[i] = 0.5 * nodal_load_mag * inertial_direc[i]
                    Felem[i+6] = Felem[i]

                elem_nodes = self.elem_conn[ielem]
                # print(f"{ielem=} {elem_nodes=}")
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                F[glob_dof] += Felem

            # add lumped mass.. in element which now contains the beam
            ielem = int(start + np.floor(mx * nelem_per_comp))
            elem_nodes = self.elem_conn[ielem]
            glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
            xi_start = mx % (1.0/nelem_per_comp)
            xi_elem = xi_start * nelem_per_comp
            Mi = Mmass * (1-xi_elem); Mf = Mmass * xi_elem
            Fmass_elem = np.zeros((12,))
            for i in range(3):
                Fmass_elem[i] = 0.5 * inertial_direc[i] * Mi * g
                Fmass_elem[i+6] = 0.5 * inertial_direc[i] * Mf * g
            F[glob_dof] += Fmass_elem

        # compute reduced forces from bcs
        # assuming we don't need to redefine self.keep_dof, defined in _build_dense_matrices
        self.Fr = F[self.keep_dof]
        return

    def _get_stresses_for_visualization(self, x):
        """ just for visualization, not VM stresses for optimization """
        # get new xpts
        lengths = x[0::7]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # init matrices
        self.stresses = np.zeros((self.ndof,))
        weights = np.zeros((self.ndof,))

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5]
            mx = x[7*icomp+6]

            # determine element orientation for first elem in comp group
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)

            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                # get the Kelem and Melem for this component
                Cfull = get_constitutive_data(self.material, t1, t2)
                CK = Cfull[:6]
                CM = Cfull[6:]

                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))

                # all 12 dof from linear static solution
                uelem = self.u[glob_dof]
                midpt_elem_stresses = np.real( get_stresses(elem_xpts, uelem, ref_axis, CK) )

                # just copy the stresses here 

                # add 50% to each of adjacent nodes (6,) => (12,) array split up
                full_elem_stresses = np.zeros((12,))
                for i in range(2):
                    full_elem_stresses[6*i:(6*i+6)] += midpt_elem_stresses

                self.stresses[glob_dof] += full_elem_stresses
                weights[glob_dof] += np.ones((12,))
            
        # now normalize global stresses by weights (this way we get stresses at junction nodes better, not just /2.0 works)
        self.stresses /= weights
        return
    
    def _get_thicknesses_for_visualization(self, x):
        """get nodal thicknesses for visualization"""
        # get new xpts
        lengths = x[0::7]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # init matrices
        self.thicknesses = np.zeros((self.nnodes,2))
        weights = np.zeros((self.nnodes,2))

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5]
            mx = x[7*icomp+6]

            # determine element orientation for first elem in comp group
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)

            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                elem_nodes = self.elem_conn[ielem]
                node1 = elem_nodes[0]; node2 = elem_nodes[1]
                self.thicknesses[node1,0] += t1
                self.thicknesses[node2,0] += t1
                self.thicknesses[node1,1] += t2
                self.thicknesses[node2,1] += t2
                weights[node1,:] += 1.0
                weights[node2,:] += 1.0
            
        # now normalize global stresses by weights (this way we get stresses at junction nodes better, not just /2.0 works)
        self.thicknesses /= weights
        return

    def get_frequencies(self, x, nmodes=5):

        # TODO : later solve gen eigenvalue problem here
        # eigenvalues, eigenvectors = eigsh(Kmat, k=2, M=Mmat, which='SM')
        print("Solving eigenvalue problem:")
        start_time = time.time()

        # solve the generalized eigenvalues problem
        if self.sparse:
            # this is bottleneck right now, speedup in a sec
            self._build_sparse_matrices(x)
            raise RuntimeError("Sparse eigenvalue solver fails to converge at the moment, may need preconditioner or fillin or reordering.. stick with dense for now")

        else: # dense
            time1 = time.time()
            self._build_dense_matrices(x)
            time2 = time.time()
            eigvals, self.eigvecs = eigh(self.Kr, self.Mr)
            time3 = time.time()
            dt_assembly = time2 - time1
            dt_solve = time3 - time2
            # print(f"{dt_assembly=} {dt_solve=}")

        self._get_thicknesses_for_visualization(x)

        # get freqs from omega^2 eigvals
        self.freqs = np.sqrt(eigvals)

        dt = time.time() - start_time
        print(f"\tsolved eigvals in {dt:.4f} seconds.")

        return self.freqs[:nmodes]

    def get_mass(self, x):
        # get mass of entire structure
        rho = self.material.rho

        # for single component, it is rho*A*L + m where A = 
        mass = 0.0
        for icomp in range(self.ncomp):
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5]
            # mx = x[7*icomp+6]

            # volume of frustum (tapered beam)
            Si = t1i * t2i 
            Sf = t1f * t2f
            V = L / 3.0 * (Si + Sf + 0.5 * (t1i * t2f + t2i * t1f))

            mass += rho * V + Mmass   
        return mass

    def solve_static(self, x):
        # TODO : add in _build_inertial_loads again..
        # computes Kr * ur = Fr in reduced system
        self._build_dense_matrices(x)
        self._build_inertial_loads(x)

        print("Solving linear static problem:")
        start_time = time.time()
        self.ur = np.linalg.solve(self.Kr, self.Fr)
        dt = time.time() - start_time
        print(f"\tsolved static in {dt:.4f} seconds.")

        self.u = np.zeros((self.ndof,))
        self.u[self.keep_dof] = self.ur[:]
        # print(F"{self.u=}")

        self._get_stresses_for_visualization(x)
        self._get_thicknesses_for_visualization(x)

        # now plot the disps of linear static?
        return self.u

    def _get_vm_fail_vec(self, x):
        # get new xpts
        lengths = x[0::7]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # init matrices
        vm_fail_vec = np.zeros((self.nelems,), dtype=np.complex128)
        self.vm_nodal = np.zeros((self.nnodes,))
        weights = np.zeros((self.nnodes,))

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            # L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            # Mmass = x[7*icomp+5]
            # mx = x[7*icomp+6]

            # determine element orientation for first elem in comp group
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)

            # get failure material data
            CS = get_stress_constitutive(self.material)

            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))

                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                # all 12 dof from linear static solution
                uelem = self.u[glob_dof]
                vm_fail_index = get_vm_stress(
                    t1, t2, elem_xpts, 
                    uelem, ref_axis, CS, 
                    self.rho_KS, self.safety_factor
                )
                vm_fail_vec[ielem] = vm_fail_index
                self.vm_nodal[elem_nodes] += np.real(vm_fail_index)
                weights[elem_nodes] += 1.0

        self.vm_nodal /= weights # normalize for visualization

        return vm_fail_vec

    def get_failure_index(self, x):
        """ just for visualization, not VM stresses for optimization """
        vm_fail_vec = self._get_vm_fail_vec(x)
        ks_fail_index = np.log(np.sum(np.exp(self.rho_KS * vm_fail_vec))) / self.rho_KS
        return ks_fail_index

    # SENSITIVITIES --------------------------

    def get_mass_gradient(self, x):
        # compute mass gradient
        mgrad = np.array([0.0]*7*self.ncomp)
        rho = self.material.rho

        for icomp in range(self.ncomp):
            # get all DVs
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            # Mmass = x[7*icomp+5]
            # mx = x[7*icomp+6]

            Si = t1i * t2i 
            Sf = t1f * t2f
            V = L / 3.0 * (Si + Sf + 0.5 * (t1i * t2f + t1f * t2i))

            Sib = rho * L / 3.0
            Sfb = rho * L / 3.0

            mgrad[7*icomp] = rho * V / L
            mgrad[7*icomp + 1] = Sib * Si / t1i + rho * L / 6.0 * t2f
            mgrad[7*icomp + 2] = Sfb * Sf / t1f + rho * L / 6.0 * t2i
            mgrad[7*icomp + 3] = Sib * Si / t2i + rho * L / 6.0 * t1f
            mgrad[7*icomp + 4] = Sfb * Sf / t2f + rho * L / 6.0 * t1i

            # lumped mass terms
            mgrad[7*icomp+5] = 1.0 # Mmass
            # mgrad[7*icomp+6] = 0.0 # mx

        return mgrad

    def get_frequency_gradient(self, x, imode):
        # get the DV gradient of natural frequencies for optimization
        freq = self.freqs[imode]
        num = self._get_dKdx_freq_term(x, imode) - freq**2 * self._get_dMdx_term(x, imode)
        den = self._get_modal_mass(imode) * 2 * freq
        return num / den

    def _get_modal_mass(self, imode):
        # phi^T * M * phi modal mass for single eigenmode with already computed M matrix
        phi_red = self.eigvecs[:,imode]
        return np.dot(np.dot(self.Mr, phi_red), phi_red)

    def get_failure_index_gradient(self, x):
        """ just for visualization, not VM stresses for optimization """

        dfail_du = self._get_dfail_du(x)
        dfail_du_red = dfail_du[self.keep_dof]
        self.psir = np.linalg.solve(self.Kr, -dfail_du_red)

        self.psi = np.zeros((self.ndof,))
        self.psi[self.keep_dof] = self.psir[:]

        dfail_dx = self._get_dfail_dx(x)
        dfail_dx += self._get_dKdx_static_term(x)
        dfail_dx -= self._get_inertial_dRdx_term(x)

        # dfail_dx[0::7] *= -1 # ? why my length derivs wrong?
        # why we need negative sign here?
        # dfail_dx *= -1.0
        return dfail_dx

    def freq_FD_test(self, x, imode, h=1e-3):
        p_vec = np.random.rand(self.num_dvs)
        self.get_frequencies(x)
        freq_grad = self.get_frequency_gradient(x, imode)
        freqs1 = self.get_frequencies(x - p_vec * h)
        freqs2 = self.get_frequencies(x + p_vec * h)
        FD_val = (freqs2[imode] - freqs1[imode]) / 2 / h
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

    def dfail_dx_FD_test(self, x, h=1e-5, idv='all'):
        # test dfail/dx grad, with u fixed (so only partial not total derivative here)
        if idv == 'all':
            p_vec = np.random.rand(self.num_dvs)
        else:
            p_vec = np.zeros((self.num_dvs))
            p_vec[idv] = 1.0

        self.solve_static(x)
        self.get_failure_index(x)
        fail_grad = self._get_dfail_dx(x)
        failn1 = self.get_failure_index(x - p_vec * h)
        fail1 = self.get_failure_index(x + p_vec * h)
        FD_val = np.real( (fail1 - failn1) / 2 / h )
        HC_val = np.dot(fail_grad, p_vec)
        print(f"dfail/dx partial grad FD test: {FD_val=} {HC_val=}")
        return

    def dfail_du_FD_test(self, x, h=1e-5):
        p_vec = np.random.rand(self.ndof)
        self.u = np.random.rand(self.ndof) * 1e-3
        fail0 = self.get_failure_index(x)
        fail_grad = self._get_dfail_du(x)
        self.u -= p_vec * h
        failn1 = self.get_failure_index(x)
        self.u += 2 * p_vec * h
        fail1 = self.get_failure_index(x)
        FD_val = np.real( (fail1 - failn1) / 2 / h )
        HC_val = np.dot(fail_grad, p_vec)
        print(f"dfail/du partial grad FD test: {FD_val=} {HC_val=}")
        return

    def fail_index_FD_test(self, x, h=1e-5):
        # test dfail/dx grad, with u fixed (so only partial not total derivative here)
        # p_vec = np.random.rand(self.num_dvs)
        p_vec = np.zeros((self.num_dvs,))
        p_vec[6+7] = 1.0 # length good now, t1 + t2 derivs fail

        self.solve_static(x)
        # self.get_failure_index(x) #fail0 = 
        fail_grad = self.get_failure_index_gradient(x)
        # fail_grad = self._get_dfail_dx(x)

        # update solve of u(x) so we account for du/dx or adjoint term
        self.solve_static(x - p_vec * h)
        failn1 = self.get_failure_index(x - p_vec * h)

        self.solve_static(x + p_vec * h)
        fail1 = self.get_failure_index(x + p_vec * h)
        FD_val = np.real( (fail1 - failn1) / 2 / h )
        HC_val = np.dot(fail_grad, p_vec)
        print(f"dfail/dx total grad FD test: {FD_val=} {HC_val=}")
        return
    
    def dKdx_FD_test(self, x, imode, h=1e-4):
        p_vec = np.random.rand(self.num_dvs)
        # p_vec[5:] = 0.0
        # p_vec = np.zeros((self.num_dvs,))
        # p_vec[6] = 1.0 # length good now, t1 + t2 derivs fail
        self.get_frequencies(x)
        # Kr0 = self.Kr.copy()
        phi = self.eigvecs[:,imode].copy()
        dKrdx_grad = self._get_dKdx_freq_term(x, imode)
        # print(F"{dKrdx_grad=}")
        # print(f"{phi[:12]=}")
        self._build_dense_matrices(x - p_vec * h)
        Kr1 = self.Kr.copy()

        self._build_dense_matrices(x + p_vec * h)
        Kr2 = self.Kr.copy()
        
        dKr_dx_p = (Kr2 - Kr1) / 2 /h
        # plt.imshow(dKr_dx_p)
        # plt.show()

        # print(F"{dKr_dx_p=}")
        # print(F"{np.diag(dKr_dx_p)=}")
        FD_val = np.dot(np.dot(dKr_dx_p, phi), phi)
        HC_val = np.dot(p_vec, dKrdx_grad)
        print(f"dK/dx FD test: {FD_val=} {HC_val=}")
        return

    def dKdx_FD_static_test(self, x, h=1e-4, idv='all'):
        if idv == 'all':
            p_vec = np.random.rand(self.num_dvs)
        else:
            p_vec = np.zeros((self.num_dvs,))
            p_vec[idv] = 1.0 # length good now, t1 + t2 derivs fail
        # NOTE : this test might miss the effect of inertial load changes
        self.solve_static(x)
        self.get_failure_index_gradient(x)
        Kr0 = self.Kr.copy()
        psir = self.psir.copy()
        ur = self.ur.copy()
        dKrdx_grad = self._get_dKdx_static_term(x)

        self.solve_static(x + p_vec * h)
        Kr2 = self.Kr.copy()
        
        dKr_dx_p = (Kr2 - Kr0) / h
        # plt.imshow(dKr_dx_p)
        # plt.show()

        # print(F"{dKr_dx_p=}")
        # print(F"{np.diag(dKr_dx_p)=}")
        FD_val = np.dot(np.dot(dKr_dx_p, psir), ur)
        HC_val = np.dot(p_vec, dKrdx_grad)
        print(f"dK/dx static FD test: {FD_val=} {HC_val=}")
        return
    
    def dRdx_inertial_FD_test(self, x, h=1e-4, idv='all'):
        if idv=='all':
            p_vec = np.random.rand(self.num_dvs)
        else:
            p_vec = np.zeros((self.num_dvs,))
            p_vec[idv] = 1.0 # length good now, t1 + t2 derivs fail

        # first just get the matrices and vec F at init design + adjoint
        self.solve_static(x)
        self.get_failure_index_gradient(x)
        # Kr0 = self.Kr.copy()
        psir = self.psir.copy()
        # ur = self.ur.copy()

        # compute the <psi,dFr/dx> HC vs FD
        dRdx_grad = self._get_inertial_dRdx_term(x)

        self._build_inertial_loads(x - p_vec * h)
        Frn1 = self.Fr.copy()

        self._build_inertial_loads(x + p_vec * h)
        Fr1 = self.Fr.copy()

        # print(f"{Frn1-Fr1=}")

        dFr_dx_p = np.dot(psir, (Fr1 - Frn1) / 2 /h)

        # print(F"{dKr_dx_p=}")
        # print(F"{np.diag(dKr_dx_p)=}")
        FD_val = dFr_dx_p
        HC_val = np.dot(p_vec, dRdx_grad)
        print(f"dR/dx inertial load (static) FD test: {FD_val=} {HC_val=}")
        return
    
    def dMdx_FD_test(self, x, imode, h=1e-5):
        p_vec = np.random.rand(self.num_dvs)
        # p_vec[5:] = 0.0
        # p_vec = np.zeros((self.num_dvs,))
        # p_vec[6] = 1.0
        self.get_frequencies(x)
        # Mr0 = self.Mr.copy()
        phi = self.eigvecs[:,imode].copy()
        dMrdx_grad = self._get_dMdx_term(x, imode)

        self._build_dense_matrices(x - p_vec * h)
        Mr1 = self.Mr.copy()

        self._build_dense_matrices(x + p_vec * h)
        Mr2 = self.Mr.copy()
        
        dMr_dx_p = (Mr2 - Mr1) / 2.0 / h
        FD_val = np.dot(np.dot(dMr_dx_p, phi), phi)
        HC_val = np.dot(p_vec, dMrdx_grad)
        print(f"dM/dx FD test: {FD_val=} {HC_val=}")
        return

    def _get_dfail_dx(self, x):
        # partial derivatives / gradient of failure index with respect to DVs\
        fail_grad = np.zeros((7 * self.ncomp,))

        lengths = x[0::7]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # KS backprop states
        vm_fail_vec = self._get_vm_fail_vec(x)
        ks_sum = np.sum(np.exp(self.rho_KS * vm_fail_vec))

        # now loop over each component computing partials --------------

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            # Mmass = x[7*icomp+5]
            # mx = x[7*icomp+6]

            # determine element orientation for first elem in comp group
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)
            elem_xpts0 = elem_xpts.copy()

            # get the Kelem and Melem for this component
            CS = get_stress_constitutive(self.material)

            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))

                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                # all 12 dof from linear static solution
                uelem = self.u[glob_dof]
                vm_fail_index = get_vm_stress(
                    t1, t2, elem_xpts, 
                    uelem, ref_axis, CS, 
                    self.rho_KS, self.safety_factor
                )

                # jacobian of vm fail index => global fail index
                ks_fail_jac = np.exp(self.rho_KS * vm_fail_index) / ks_sum

                # compute dvm/dL term ----------------
                h = 1e-30 # complex-step method on first order
                zero = np.array([0.0]*3)
                pert_xpts = np.concatenate([zero, np.abs(dxpt) / L], axis=0)
                dvm_dL = np.imag(get_vm_stress(
                    t1, t2, elem_xpts0 + pert_xpts * 1j * h,
                    uelem, ref_axis, CS,
                    self.rho_KS, self.safety_factor,
                )) / h * 2.0
                fail_grad[7*icomp] += dvm_dL * np.real(ks_fail_jac)

                # compute dvm/dt1 term ----------------
                dvm_dt1 = np.imag(get_vm_stress(
                    t1 + h * 1j, t2, elem_xpts0,
                    uelem, ref_axis, CS,
                    self.rho_KS, self.safety_factor,
                )) / h * 2.0
                dvm_dt1 *= np.real(ks_fail_jac)
                
                # back to t1i and t1f (mass doesn't affect this xi, this is elem centroid)
                fail_grad[7*icomp + 1] += dvm_dt1 * (1-xi)
                fail_grad[7*icomp + 2] += dvm_dt1 * xi

                # compute dvm/dL term ----------------
                dvm_dt2 = np.imag(get_vm_stress(
                    t1, t2 + h * 1j, elem_xpts0,
                    uelem, ref_axis, CS,
                    self.rho_KS, self.safety_factor,
                )) / h * 2.0
                dvm_dt2 *= np.real(ks_fail_jac)
                
                # back to t2i and t2f (mass doesn't affect this xi, this is elem centroid)
                fail_grad[7*icomp + 3] += dvm_dt2 * (1-xi)
                fail_grad[7*icomp + 4] += dvm_dt2 * xi

         # not sure why /2.0 here have 2.0 above thought that was right
         # maybe because we add twice? also not sure why negative, but that mathces better
        return fail_grad / 2.0 * -1.0

    def _get_dfail_du(self, x):
        # partial derivatives / gradient of failure index with respect to disp states
        fail_gradu = np.zeros((self.ndof,))

        lengths = x[0::7]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # KS backprop states
        vm_fail_vec = self._get_vm_fail_vec(x)
        ks_sum = np.sum(np.exp(self.rho_KS * vm_fail_vec))

        # now loop over each component computing partials --------------

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5]
            mx = x[7*icomp+6]

            # determine element orientation for first elem in comp group
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)

            # get the Kelem and Melem for this component
            CS = get_stress_constitutive(self.material)

            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))

                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                # all 12 dof from linear static solution
                uelem = self.u[glob_dof]
                vm_fail_index = get_vm_stress(
                    t1, t2, elem_xpts, 
                    uelem, ref_axis, CS, 
                    self.rho_KS, self.safety_factor
                )

                # jacobian of vm fail index => global fail index
                ks_fail_jac = np.exp(self.rho_KS * vm_fail_index) / ks_sum

                # now compute dvm_fail/duelem at element level here
                dvm_duelem = np.zeros((12,))
                h = 1e-30
                for i in range(12):
                    pert_uelem = np.zeros((12,))
                    pert_uelem[i] = 1.0
                    dvm_duelem[i] = np.imag(get_vm_stress(
                        t1, t2, elem_xpts, 
                        uelem + pert_uelem * 1j * h, ref_axis, CS, 
                        self.rho_KS, self.safety_factor
                    )) / h * 2.0

                fail_gradu[glob_dof] += dvm_duelem * np.real(ks_fail_jac)

         # not sure why /2.0 here have 2.0 above thought that was right
         # maybe because we add twice?
        return fail_gradu / 2.0

    def _get_dKdx_static_term(self, x):
        # use trick to get psi^T * dK/dx * u
        # uses strain energy bidirec term (since linear strains and quadratic U to get without forming Kelem)
        dKdx_grad = np.zeros((7 * self.ncomp,))
        nelem_per_comp = self.tree.nelem_per_comp

        # get relevant eigenvector
        phi_red = self.psir
        phi = np.zeros((self.ndof,))
        phi[self.keep_dof] = phi_red[:]

        # loop over each component to get local DV derivs
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5]
            mx = x[7*icomp+6]

            # prelim --------------
            # get orient ind and ref axis
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0
            elem_xpts0 = np.concatenate([xpt1, xpt2], axis=0)

            # now do assembly step for each element -----
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                phie = phi[glob_dof]
                ue = self.u[glob_dof]

                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                # initial const data
                Cfull = get_constitutive_data(self.material, t1, t2)
                CK = Cfull[:6]; CM = Cfull[6:]

                # compute dK/dL term ------------
                h = 1e-30 # complex-step method on first order
                zero = np.array([0.0]*3)
                pert_xpts = np.concatenate([zero, np.abs(dxpt) / L], axis=0)
                dKdx_grad[7*icomp] += np.imag(get_bidirec_strain_energy(
                    elem_xpts0 + pert_xpts * 1j * h,
                    phie, ue, ref_axis, CK
                )) / h * 2.0

                # print(f"{dKdx_grad[7*icomp]=}")

                # compute dK/t1 term ------------
                Cd1 = get_constitutive_data(self.material, t1 + h * 1j, t2)
                dKdt1 = np.imag(get_bidirec_strain_energy(
                    elem_xpts0,
                    phie, ue, ref_axis,
                    Cd1[:6]
                )) / h * 2.0

                # back to t1i and t1f (mass doesn't affect this xi, this is elem centroid)
                dKdx_grad[7*icomp + 1] += dKdt1 * (1-xi)
                dKdx_grad[7*icomp + 2] += dKdt1 * xi

                # compute dK/t2 term ------------
                Cd2 = get_constitutive_data(self.material, t1, t2 + h * 1j)
                dKdt2 = np.imag(get_bidirec_strain_energy(
                    elem_xpts0,
                    phie, ue, ref_axis,
                    Cd2[:6]
                )) / h * 2.0

                # back to t1i and t1f (mass doesn't affect this xi, this is elem centroid)
                dKdx_grad[7*icomp + 3] += dKdt2 * (1-xi)
                dKdx_grad[7*icomp + 4] += dKdt2 * xi

                # no lumped mass terms influence Kmat, so zero derivs here

        return dKdx_grad

    def _get_inertial_dRdx_term(self, x):
        # TODO : compute inertial loads n * rho * g * A on different parts of the structure..
        assert(self.inertial_data)
        # this inertial grad is <psi, dfext/dx> doesn't have neg sign on it
        inertial_grad = np.zeros((7 * self.ncomp,))

        # assuming constant load factor for now, not differentiated
        mass = self.get_mass(x)     
        nelem_per_comp = self.tree.nelem_per_comp
        # n = self.inertial_data.get_load_factor(mass) # TODO : later
        inertial_direc = self.inertial_data.inertial_direction
        rho = self.material.rho
        g = self.inertial_data.accel_grav

        # get static analysis adjoint vector for failure index
        psi_red = self.psir
        psi = np.zeros((self.ndof,))
        psi[self.keep_dof] = psi_red[:]

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5]
            mx = x[7*icomp+6]

            # compute element length Le
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            Le = np.linalg.norm(xpt2 - xpt1)

            # now we compute derivs (looping over each element on the fly)
            # ---------------------

            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                # print(f"{ielem=} {elem_nodes=}")
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))

                # get the xi [0,1] at start and end of element
                xi1 = ielem/nelem_per_comp
                xi2 = xi1 + 1.0 / nelem_per_comp
                t11 = t1i * (1-xi1) + t1f * xi1
                t12 = t1i * (1-xi2) + t1f * xi2
                t21 = t2i * (1-xi1) + t2f * xi1
                t22 = t2i * (1-xi2) + t2f * xi2
                S1 = t11 * t21 # CS area at start of element
                S2 = t12 * t22 # CS area at end of element
                # volume of frustum (tapered beam section)
                V = Le / 3.0 * (S1 + S2 + 0.5 * (t11 * t22 + t12 * t21))

                nodal_load_mag = rho * g * V
                # print(f"{nodal_load_mag=}")

                # don't actually need beam direc, just need to compute comp x,y,z parts of distributed load
                # compute element nodal loads
                Felem = np.zeros((12,))
                for i in range(3):
                    # this is not perfect, could distribute mass better, but it's fine for now probably
                    # just 0.5 to each node
                    Felem[i] = 0.5 * nodal_load_mag * inertial_direc[i]
                    Felem[i+6] = Felem[i]

                # get element adjoint vector
                psie = psi[glob_dof]

                # compute some prelim scalar derivatives first
                dmag_dV = nodal_load_mag / V
                dmag_dS1 = dmag_dV * Le / 3.0
                dmag_dS2 = dmag_dV * Le / 3.0
                dmag_dt1i = dmag_dS1 * t21 * (1 - xi1) + dmag_dS2 * t22 * (1 - xi2) + dmag_dV * Le/6.0 * (t22 * (1-xi1) + t21 * (1-xi2))
                dmag_dt1f = dmag_dS1 * t21 * xi1 + dmag_dS2 * t22 * xi2 + dmag_dV * Le/6.0 * (t22 * xi1 + t21 * xi2)
                dmag_dt2i = dmag_dS1 * t11 * (1 - xi1) + dmag_dS2 * t12 * (1 - xi2) + dmag_dV * Le/6.0 * (t12 * (1-xi1) + t11 * (1-xi2))
                dmag_dt2f = dmag_dS1 * t11 * xi1 + dmag_dS2 * t12 * xi2 + dmag_dV * Le/6.0 * (t12 * xi1 + t11 * xi2)

                # first compute L deriv
                dFelem_dL = Felem / L
                inertial_grad[7*icomp] += np.dot(psie, dFelem_dL)

                # compute t1i deriv
                dFelem_dt1i = Felem / nodal_load_mag * dmag_dt1i
                inertial_grad[7*icomp + 1] += np.dot(psie, dFelem_dt1i)

                # compute t1f deriv
                dFelem_dt1f = Felem / nodal_load_mag * dmag_dt1f
                inertial_grad[7*icomp + 2] += np.dot(psie, dFelem_dt1f)

                # compute t2i deriv
                dFelem_dt2i = Felem / nodal_load_mag * dmag_dt2i
                inertial_grad[7*icomp + 3] += np.dot(psie, dFelem_dt2i)

                # compute t2f deriv
                dFelem_dt2f = Felem / nodal_load_mag * dmag_dt2f
                inertial_grad[7*icomp + 4] += np.dot(psie, dFelem_dt2f)

            # lumped mass derivs
            ielem = int(start + np.floor(mx * nelem_per_comp))
            elem_nodes = self.elem_conn[ielem]
            glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
            psie = psi[glob_dof]
            xi_start = mx % (1.0/nelem_per_comp)
            xi_elem = xi_start * nelem_per_comp
            Mi = Mmass * (1-xi_elem); Mf = Mmass * xi_elem
            Fmass_elem = np.zeros((12,))
            for i in range(3):
                Fmass_elem[i] = 0.5 * inertial_direc[i] * Mi * g
                Fmass_elem[i+6] = 0.5 * inertial_direc[i] * Mf * g
            
            dFmass_dmx = np.zeros((12,))
            dMi_dmx = -Mmass * nelem_per_comp
            dMf_dmx = Mmass * nelem_per_comp
            for i in range(3):
                dFmass_dmx[i] = 0.5 * inertial_direc[i] * dMi_dmx * g
                dFmass_dmx[i+6] = 0.5 * inertial_direc[i] * dMf_dmx * g

            inertial_grad[7*icomp + 5] += np.dot(psie, Fmass_elem / Mmass)
            inertial_grad[7*icomp + 6] += np.dot(psie, dFmass_dmx)

        return inertial_grad

    def _get_dKdx_freq_term(self, x, imode):
        # use trick to get phi^T * dK/dx * phi
        # namely find dU/dx at element level with Ue(phi,x) of ue^T * Ke * ue with ue = phi\
        dKdx_grad = np.zeros((7 * self.ncomp,))
        nelem_per_comp = self.tree.nelem_per_comp

        # get relevant eigenvector
        phi_red = self.eigvecs[:,imode]
        phi = np.zeros((self.ndof,))
        phi[self.keep_dof] = phi_red[:]

        # loop over each component to get local DV derivs
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            # Mmass = x[7*icomp+5]
            # mx = x[7*icomp+6]

            # prelim --------------
            # get orient ind and ref axis
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0
            elem_xpts0 = np.concatenate([xpt1, xpt2], axis=0)

            # now do assembly step for each element -----
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                phie = phi[glob_dof]

                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                # initial const data
                Cfull = get_constitutive_data(self.material, t1, t2)
                CK = Cfull[:6]; CM = Cfull[6:]

                # compute dK/dL term ------------
                h = 1e-30 # complex-step method on first order
                zero = np.array([0.0]*3)
                pert_xpts = np.concatenate([zero, np.abs(dxpt) / L], axis=0)
                dKdx_grad[7*icomp] += np.imag(get_strain_energy(
                    elem_xpts0 + pert_xpts * 1j * h,
                    phie, ref_axis, CK
                )) / h * 2.0

                # compute dK/t1 term ------------
                Cd1 = get_constitutive_data(self.material, t1 + h * 1j, t2)
                dKdt1 = np.imag(get_strain_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd1[:6]
                )) / h * 2.0

                # back to t1i and t1f (mass doesn't affect this xi, this is elem centroid)
                dKdx_grad[7*icomp + 1] += dKdt1 * (1-xi)
                dKdx_grad[7*icomp + 2] += dKdt1 * xi

                # compute dK/t2 term ------------
                Cd2 = get_constitutive_data(self.material, t1, t2 + h * 1j)
                dKdt2 = np.imag(get_strain_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd2[:6]
                )) / h * 2.0

                # back to t1i and t1f (mass doesn't affect this xi, this is elem centroid)
                dKdx_grad[7*icomp + 3] += dKdt2 * (1-xi)
                dKdx_grad[7*icomp + 4] += dKdt2 * xi

        return dKdx_grad

    def _get_dMdx_term(self, x, imode):
        # use trick to get phi^T * dM/dx * phi
        # namely find dU/dx at element level with Te(phi,x) of ue^T * Me * ue with ue = phi
        dMdx_grad = np.zeros((7 * self.ncomp,))
        nelem_per_comp = self.tree.nelem_per_comp

        # get relevant eigenvector
        phi_red = self.eigvecs[:,imode]
        phi = np.zeros((self.ndof,))
        phi[self.keep_dof] = phi_red[:]

        # loop over each component to get local DV derivs
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[7*icomp]
            t1i = x[7*icomp+1]
            t1f = x[7*icomp+2]
            t2i = x[7*icomp+3]
            t2f = x[7*icomp+4]
            Mmass = x[7*icomp+5]
            mx = x[7*icomp+6]

            # prelim --------------
            # get orient ind and ref axis
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            dxpt = xpt2 - xpt1
            orient_ind = np.argmax(np.abs(dxpt))
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0
            elem_xpts0 = np.concatenate([xpt1, xpt2], axis=0)

            # now do assembly step for each element -----
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                phie = phi[glob_dof]

                # compute the midpoint of the element on 0 to 1 coordinate
                xi = (ielem-start+1)/nelem_per_comp - 0.5 / nelem_per_comp

                # compute the thicknesses at this element
                t1 = t1i * (1-xi) + t1f * xi
                t2 = t2i * (1-xi) + t2f * xi

                # initial const data
                Cfull = get_constitutive_data(self.material, t1, t2)
                CK = Cfull[:6]; CM = Cfull[6:]

                # compute dK/dL term ------------
                h = 1e-30 # complex-step method on first order
                zero = np.array([0.0]*3)
                pert_xpts = np.concatenate([zero, np.abs(dxpt) / L], axis=0)
                dMdx_grad[7*icomp] += np.imag(get_kinetic_energy(
                    elem_xpts0 + pert_xpts * 1j * h,
                    phie, ref_axis, CM
                )) / h * 2.0

                # compute dM/t1 term ------------
                Cd1 = get_constitutive_data(self.material, t1 + h * 1j, t2)
                dMdt1 = np.imag(get_kinetic_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd1[6:]
                )) / h * 2.0

                # back to t1i and t1f (mass doesn't affect this xi, this is elem centroid)
                dMdx_grad[7*icomp + 1] += dMdt1 * (1-xi)
                dMdx_grad[7*icomp + 2] += dMdt1 * xi

                # compute dM/t2 term ------------
                Cd2 = get_constitutive_data(self.material, t1, t2 + h * 1j)
                dMdt2 = np.imag(get_kinetic_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd2[6:]
                )) / h * 2.0

                # back to t1i and t1f (mass doesn't affect this xi, this is elem centroid)
                dMdx_grad[7*icomp + 3] += dMdt2 * (1-xi)
                dMdx_grad[7*icomp + 4] += dMdt2 * xi

            # get lumped mass derivs..in which elem contains the beam
            ielem = int(start + np.floor(mx * nelem_per_comp))
            elem_nodes = self.elem_conn[ielem]
            glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
            phie = phi[glob_dof]
            xi_start = mx % (1.0/nelem_per_comp)
            # xi_elem = xi_start * nelem_per_comp
            # Mi = Mmass * (1-xi_elem); Mf = Mmass * xi_elem
            # Melem_lumped = np.diag(Mi * [1]*6 + Mf * [1]*6) / 2.0
            dM_dMi = np.dot(phie[:6], phie[:6]) / 2.0
            dM_dMf = np.dot(phie[6:], phie[6:]) / 2.0 # /2 bc half above, /2 again bc KE

            # lumped mass gradients
            dMdx_grad[7*icomp + 5] += (dM_dMi + dM_dMf) / 2.0
            dMi_dmx = -Mmass * nelem_per_comp
            dMf_dmx = Mmass * nelem_per_comp
            dMdx_grad[7*icomp + 6] += dM_dMi * dMi_dmx + dM_dMf * dMf_dmx

        return dMdx_grad

    # PLOT UTILS -------------------------

    def write_freq_to_vtk(self, nmodes:int, file_prefix:str=""):
        """writes to vtk"""
        for imode in range(nmodes):
            # get full eigenmode shape
            phi_r = self.eigvecs[:,imode]
            phi = np.zeros((self.ndof,))
            phi[self.keep_dof] = phi_r

            write_beam_modes_to_vtk(
                filename=f"{file_prefix}_mode{imode}.vtk", 
                node_coords=np.reshape(self.xpts, newshape=(self.nnodes, 3)),
                elements=np.reshape(np.array(self.elem_conn), newshape=(self.nelems, 2)),
                mode_shapes=np.reshape(phi, newshape=(self.nnodes, 6)),
                thicknesses=np.reshape(self.thicknesses, newshape=(self.nnodes,2))
            )
        return

    def write_static_to_vtk(self, file_prefix:str=""):
        """writes to vtk"""

        write_beam_static_to_vtk(
            filename=f"{file_prefix}_static.vtk", 
            node_coords=np.reshape(self.xpts, newshape=(self.nnodes, 3)), 
            elements=np.reshape(np.array(self.elem_conn), newshape=(self.nelems, 2)), 
            disps=np.reshape(self.u, newshape=(self.nnodes, 6)), 
            stresses=np.reshape(self.stresses, newshape=(self.nnodes, 6)), # TODO
            vm_stress=self.vm_nodal, # TODO
            thicknesses=np.reshape(self.thicknesses, newshape=(self.nnodes,2))
        )
        return

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