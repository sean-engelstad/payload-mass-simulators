__all__ = ["Material", "TreeData", "BeamAssembler"]

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
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, spilu
from .sparse_utils import apply_rcm_reordering

class BeamAssembler:
    def __init__(self, material:Material, tree:TreeData, inertial_data:InertialData=None, rho_KS:float=10.0, safety_factor:float=1.5, sparse:bool=False):
        self.material = material
        self.tree = tree
        self.inertial_data = inertial_data
        self.rho_KS = rho_KS
        self.safety_factor = safety_factor
        self.sparse = sparse

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

        self.tree.ndvs_per_comp = 3

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

        self.freq_err_hist = []
        self.freq_hist = []

        # any prelim
        # ----------

        # only need to do this one time
        if self.sparse:
            self._compute_nz_pattern()

    @classmethod
    def create_sparse(
        cls,
        material:Material, 
        tree:TreeData, 
        inertial_data:InertialData=None, 
        rho_KS:float=10.0, 
        safety_factor:float=1.5
    ):
        return cls(material, tree, inertial_data, rho_KS, safety_factor, sparse=True)
    
    @classmethod
    def create_dense(
        cls,
        material:Material, 
        tree:TreeData, 
        inertial_data:InertialData=None, 
        rho_KS:float=10.0, 
        safety_factor:float=1.5
    ):
        return cls(material, tree, inertial_data, rho_KS, safety_factor, sparse=False)

    def _build_dense_matrices(self, x):
        """sparse matrices available, but dense eigval solver more robust atm"""

        # get new xpts
        lengths = x[0::3]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # init matrices
        K = np.zeros((self.ndof, self.ndof))
        M = np.zeros((self.ndof, self.ndof))

        tot_dt = 0.0

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
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)
            # switch to xpts in x-dir then permute x,y,z
            # elem_xpts = np.array([0.0] * 3 + [L/nelem_per_comp, 0.0, 0.0])
            qvars = np.array([0.0]*12)

            # get the Kelem and Melem for this component
            time1 = time.time()
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]
            CM = Cfull[6:]
            time15 = time.time()
            Kelem = get_stiffness_matrix(elem_xpts, qvars, ref_axis, CK)
            time16 = time.time()
            Melem = get_mass_matrix(elem_xpts, qvars, ref_axis, CM)
            time2 = time.time()
            dtK = time16 - time15
            dtM = time2 - time16
            dt = time2 - time1
            # print(f"{dt=}\t{dtK=}\t{dtM=}") # time debugging
            tot_dt += dt

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

        # apply bcs for reduced matrix --------------
        bcs = [_ for _ in range(6)]
        self.keep_dof = [_ for _ in range(self.ndof) if not(_ in bcs)]
        self.Kr = K[self.keep_dof,:][:,self.keep_dof]
        self.Mr = M[self.keep_dof,:][:,self.keep_dof]

        print(f"{tot_dt=}")

        # plt.imshow(self.Kr)
        # plt.show()

    def _build_inertial_loads(self, x):
        # TODO : compute inertial loads n * rho * g * A on different parts of the structure..
        assert(self.inertial_data)

        # assuming constant load factor for now, not differentiated
        mass = self.get_mass(x)     
        nelem_per_comp = self.tree.nelem_per_comp
        n = self.inertial_data.get_load_factor(mass)
        inertial_direc = self.inertial_data.inertial_direction
        rho = self.material.rho
        g = self.inertial_data.accel_grav

        # assemble global forces
        F = np.zeros((self.ndof,)) # global force vector

        # loop over components and elements
        for icomp in range(self.ncomp):
            # local des vars in this component
            L = x[3*icomp]
            t1 = x[3*icomp+1]
            t2 = x[3*icomp+2]

            # compute element length Le
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            Le = np.linalg.norm(xpt2 - xpt1)

            # compute distr load mag based on element vs inertial orientation
            # and the base distr load mag
            A = t1 * t2
            distr_load_mag = rho * g * A
            nodal_load_mag = distr_load_mag * Le # total load on beam section
            # print(f"{nodal_load_mag=}")

            # don't actually need beam direc, just need to compute comp x,y,z parts of distributed load
            # compute element nodal loads
            Felem = np.zeros((12,))
            for i in range(3):
                cart_direc = np.zeros((3,))
                Felem[i] = 0.5 * nodal_load_mag * inertial_direc[i]
                Felem[i+6] = Felem[i]

            # now do assembly step for each element
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                # print(f"{ielem=} {elem_nodes=}")
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                F[glob_dof] += Felem

        # compute reduced forces from bcs
        # assuming we don't need to redefine self.keep_dof, defined in _build_dense_matrices
        self.Fr = F[self.keep_dof]
        return

    def _get_stresses_for_visualization(self, x):
        """ just for visualization, not VM stresses for optimization """
        # get new xpts
        lengths = x[0::3]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # init matrices
        self.stresses = np.zeros((self.ndof,))
        weights = np.zeros((self.ndof,))

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
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)

            # get the Kelem and Melem for this component
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]
            CM = Cfull[6:]

            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
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
            # doesn't converge, badly ill-conditioned
            eigvals, self.eigvecs = eigsh(self.Kr, nmodes, self.Mr, which='SM')
            # # shift and invert approach fails (below is how you might do preconditioner)
            # Kr_csr = self.Kr.tocsr()
            # Mr_csr = self.Mr.tocsr()
            # Kr_csr, perm = apply_rcm_reordering(Kr_csr)
            # M = spilu(Kr_csr.tocsc())
            # preconditioner = LinearOperator(shape=Kr_csr.shape, matvec=M.solve)
            # eigvals, self.eigvecs = eigsh(Kr_csr, k=nmodes, M=Mr_csr, which='SM', maxiter=10000, tol=1e-5, OPinv=preconditioner, sigma=2.0)

        else: # dense
            # time1 = time.time()
            self._build_dense_matrices(x)
            # time2 = time.time()
            eigvals, self.eigvecs = eigh(self.Kr, self.Mr)
            # time3 = time.time()
            # dt_assembly = time2 - time1
            # dt_solve = time3 - time2
            # print(f"{dt_assembly=} {dt_solve=}")

        # get freqs from omega^2 eigvals
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

        # now plot the disps of linear static?
        return self.u

    def _get_vm_fail_vec(self, x):
        # get new xpts
        lengths = x[0::3]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # init matrices
        vm_fail_vec = np.zeros((self.nelems,), dtype=np.complex128)
        self.vm_nodal = np.zeros((self.nnodes,))
        weights = np.zeros((self.nnodes,))

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
        # why we need negative sign here?
        dfail_dx *= -1.0
        return dfail_dx

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

    def dfail_dx_FD_test(self, x, h=1e-5):
        # test dfail/dx grad, with u fixed (so only partial not total derivative here)
        p_vec = np.random.rand(self.num_dvs)
        self.solve_static(x)
        fail0 = self.get_failure_index(x)
        fail_grad = self._get_dfail_dx(x)
        fail1 = self.get_failure_index(x + p_vec * h)
        FD_val = np.real( (fail1 - fail0) / h )
        HC_val = np.dot(fail_grad, p_vec)
        print(f"dfail/dx partial grad FD test: {FD_val=} {HC_val=}")
        return

    def dfail_du_FD_test(self, x, h=1e-5):
        p_vec = np.random.rand(self.ndof)
        self.u = np.random.rand(self.ndof) * 1e-3
        fail0 = self.get_failure_index(x)
        fail_grad = self._get_dfail_du(x)
        self.u += p_vec * h
        fail1 = self.get_failure_index(x)
        FD_val = np.real( (fail1 - fail0) / h )
        HC_val = np.dot(fail_grad, p_vec)
        print(f"dfail/du partial grad FD test: {FD_val=} {HC_val=}")
        return

    def fail_index_FD_test(self, x, h=1e-5):
        # test dfail/dx grad, with u fixed (so only partial not total derivative here)
        p_vec = np.random.rand(self.num_dvs)
        self.solve_static(x)
        fail0 = self.get_failure_index(x)
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
        # p_vec = np.zeros((self.num_dvs,))
        # p_vec[2] = 1.0 # length good now, t1 + t2 derivs fail
        self.get_frequencies(x)
        Kr0 = self.Kr.copy()
        phi = self.eigvecs[:,imode].copy()
        dKrdx_grad = self._get_dKdx_freq_term(x, imode)
        # print(F"{dKrdx_grad=}")
        # print(f"{phi[:12]=}")
        self.get_frequencies(x + p_vec * h)
        Kr2 = self.Kr.copy()
        
        dKr_dx_p = (Kr2 - Kr0) / h
        # plt.imshow(dKr_dx_p)
        # plt.show()

        # print(F"{dKr_dx_p=}")
        # print(F"{np.diag(dKr_dx_p)=}")
        FD_val = np.dot(np.dot(dKr_dx_p, phi), phi)
        HC_val = np.dot(p_vec, dKrdx_grad)
        print(f"dK/dx FD test: {FD_val=} {HC_val=}")
        return

    def dKdx_FD_static_test(self, x, h=1e-4):
        p_vec = np.random.rand(self.num_dvs)
        # p_vec = np.zeros((self.num_dvs,))
        # p_vec[2] = 1.0 # length good now, t1 + t2 derivs fail
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
    
    def dMdx_FD_test(self, x, imode, h=1e-5):
        p_vec = np.random.rand(self.num_dvs)
        # p_vec = np.zeros((self.num_dvs,))
        # p_vec[0] = 1.0
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

    def _get_dfail_dx(self, x):
        # partial derivatives / gradient of failure index with respect to DVs\
        fail_grad = np.zeros((3 * self.ncomp,))

        lengths = x[0::3]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # KS backprop states
        vm_fail_vec = self._get_vm_fail_vec(x)
        ks_sum = np.sum(np.exp(self.rho_KS * vm_fail_vec))

        # now loop over each component computing partials --------------

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
                fail_grad[3*icomp] += dvm_dL * np.real(ks_fail_jac)

                # compute dvm/dt1 term ----------------
                dvm_dt1 = np.imag(get_vm_stress(
                    t1 + h * 1j, t2, elem_xpts0,
                    uelem, ref_axis, CS,
                    self.rho_KS, self.safety_factor,
                )) / h * 2.0
                fail_grad[3*icomp + 1] += dvm_dt1 * np.real(ks_fail_jac)

                # compute dvm/dL term ----------------
                dvm_dt2 = np.imag(get_vm_stress(
                    t1, t2 + h * 1j, elem_xpts0,
                    uelem, ref_axis, CS,
                    self.rho_KS, self.safety_factor,
                )) / h * 2.0
                fail_grad[3*icomp + 2] += dvm_dt2 * np.real(ks_fail_jac)

         # not sure why /2.0 here have 2.0 above thought that was right
         # maybe because we add twice? also not sure why negative, but that mathces better
        return fail_grad / 2.0 * -1.0

    def _get_dfail_du(self, x):
        # partial derivatives / gradient of failure index with respect to disp states
        fail_gradu = np.zeros((self.ndof,))

        lengths = x[0::3]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        # KS backprop states
        vm_fail_vec = self._get_vm_fail_vec(x)
        ks_sum = np.sum(np.exp(self.rho_KS * vm_fail_vec))

        # now loop over each component computing partials --------------

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
        dKdx_grad = np.zeros((3 * self.ncomp,))
        nelem_per_comp = self.tree.nelem_per_comp

        # get relevant eigenvector
        phi_red = self.psir
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
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0
            elem_xpts0 = np.concatenate([xpt1, xpt2], axis=0)

            # initial const data
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]; CM = Cfull[6:]

            # now do assembly step for each element -----
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                phie = phi[glob_dof]
                ue = self.u[glob_dof]

                # compute dK/dL term ------------
                h = 1e-30 # complex-step method on first order
                zero = np.array([0.0]*3)
                pert_xpts = np.concatenate([zero, np.abs(dxpt) / L], axis=0)
                dKdx_grad[3*icomp] += np.imag(get_bidirec_strain_energy(
                    elem_xpts0 + pert_xpts * 1j * h,
                    phie, ue, ref_axis, CK
                )) / h * 2.0

                # compute dK/t1 term ------------
                Cd1 = get_constitutive_data(self.material, t1 + h * 1j, t2)
                dKdx_grad[3*icomp+1] += np.imag(get_bidirec_strain_energy(
                    elem_xpts0,
                    phie, ue, ref_axis,
                    Cd1[:6]
                )) / h * 2.0

                # compute dK/t2 term ------------
                Cd2 = get_constitutive_data(self.material, t1, t2 + h * 1j)
                dKdx_grad[3*icomp+2] += np.imag(get_bidirec_strain_energy(
                    elem_xpts0,
                    phie, ue, ref_axis,
                    Cd2[:6]
                )) / h * 2.0
        return dKdx_grad

    def _get_inertial_dRdx_term(self, x):
        # TODO : compute inertial loads n * rho * g * A on different parts of the structure..
        assert(self.inertial_data)
        # this inertial grad is <psi, dfext/dx> doesn't have neg sign on it
        inertial_grad = np.zeros((3 * self.ncomp,))

        # assuming constant load factor for now, not differentiated
        mass = self.get_mass(x)     
        nelem_per_comp = self.tree.nelem_per_comp
        n = self.inertial_data.get_load_factor(mass)
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
            L = x[3*icomp]
            t1 = x[3*icomp+1]
            t2 = x[3*icomp+2]

            # compute element length Le
            first_elem = nelem_per_comp * icomp
            nodes = self.elem_conn[first_elem]
            node1 = nodes[0]; node2 = nodes[1]
            xpt1 = self.xpts[3*node1:3*node1+3]; xpt2 = self.xpts[3*node2:3*node2+3]
            Le = np.linalg.norm(xpt2 - xpt1)

            # baseline load mag
            A = t1 * t2
            distr_load_mag = rho * g * A
            nodal_load_mag = distr_load_mag * Le # total load on beam section

            # baseline Felem
            # don't actually need beam direc, just need to compute comp x,y,z parts of distributed load
            # compute element nodal loads
            Felem = np.zeros((12,))
            for i in range(3):
                cart_direc = np.zeros((3,))
                Felem[i] = 0.5 * nodal_load_mag * inertial_direc[i]
                Felem[i+6] = Felem[i]

            # now we compute derivs (looping over each element on the fly)
            # ---------------------

            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                # print(f"{ielem=} {elem_nodes=}")
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))

                # get element adjoint vector
                psie = psi[glob_dof]

                # first compute L deriv
                dFelem_dL = Felem / L
                inertial_grad[3*icomp] += np.dot(psie, dFelem_dL)

                # second compute t1 deriv
                dFelem_dt1 = Felem / t1
                inertial_grad[3*icomp + 1] += np.dot(psie, dFelem_dt1)

                # third compute t2 deriv
                dFelem_dt2 = Felem / t2
                inertial_grad[3*icomp + 2] += np.dot(psie, dFelem_dt2)

        return inertial_grad

    def _get_dKdx_freq_term(self, x, imode):
        # use trick to get phi^T * dK/dx * phi
        # namely find dU/dx at element level with Ue(phi,x) of ue^T * Ke * ue with ue = phi\
        dKdx_grad = np.zeros((3 * self.ncomp,))
        nelem_per_comp = self.tree.nelem_per_comp

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
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0
            elem_xpts0 = np.concatenate([xpt1, xpt2], axis=0)

            # initial const data
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]; CM = Cfull[6:]

            # now do assembly step for each element -----
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                phie = phi[glob_dof]

                # compute dK/dL term ------------
                h = 1e-30 # complex-step method on first order
                zero = np.array([0.0]*3)
                pert_xpts = np.concatenate([zero, np.abs(dxpt) / L], axis=0)
                dKdx_grad[3*icomp] += np.imag(get_strain_energy(
                    elem_xpts0 + pert_xpts * 1j * h,
                    phie, ref_axis, CK
                )) / h * 2.0

                # compute dK/t1 term ------------
                Cd1 = get_constitutive_data(self.material, t1 + h * 1j, t2)
                dKdx_grad[3*icomp+1] += np.imag(get_strain_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd1[:6]
                )) / h * 2.0

                # compute dK/t2 term ------------
                Cd2 = get_constitutive_data(self.material, t1, t2 + h * 1j)
                dKdx_grad[3*icomp+2] += np.imag(get_strain_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd2[:6]
                )) / h * 2.0
        return dKdx_grad

    def _get_dMdx_term(self, x, imode):
        # use trick to get phi^T * dM/dx * phi
        # namely find dU/dx at element level with Te(phi,x) of ue^T * Me * ue with ue = phi
        dMdx_grad = np.zeros((3 * self.ncomp,))
        nelem_per_comp = self.tree.nelem_per_comp

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
            rem_orient_ind = np.array([_ for _ in range(3) if not(_ == orient_ind)])
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0
            elem_xpts0 = np.concatenate([xpt1, xpt2], axis=0)

            # initial const data
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]; CM = Cfull[6:]

            # now do assembly step for each element -----
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                glob_dof = np.sort(np.array([6*inode+_ for _ in range(6) for inode in elem_nodes]))
                phie = phi[glob_dof]

                # compute dK/dL term ------------
                h = 1e-30 # complex-step method on first order
                zero = np.array([0.0]*3)
                pert_xpts = np.concatenate([zero, np.abs(dxpt) / L], axis=0)
                dMdx_grad[3*icomp] += np.imag(get_kinetic_energy(
                    elem_xpts0 + pert_xpts * 1j * h,
                    phie, ref_axis, CM
                )) / h * 2.0

                # compute dK/t1 term ------------
                Cd1 = get_constitutive_data(self.material, t1 + h * 1j, t2)
                dMdx_grad[3*icomp+1] += np.imag(get_kinetic_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd1[6:]
                )) / h * 2.0

                # compute dK/t2 term ------------
                Cd2 = get_constitutive_data(self.material, t1, t2 + h * 1j)
                dMdx_grad[3*icomp+2] += np.imag(get_kinetic_energy(
                    elem_xpts0,
                    phie, ref_axis,
                    Cd2[6:]
                )) / h * 2.0
        return dMdx_grad
    
    # PROTOTYPE SPARSE VERSION OF CODE --------------

    def _compute_nz_pattern(self):
        """ compute the nonzero pattern of the Kmat and Mmat"""

        # construct the nonzero or nodal sparsity pattern of the BSR matrix
        # one fewer as node 0 is constrained and we construct reduced mat here
        matrix_shape = (6 * (self.nnodes - 1), 6 * (self.nnodes - 1))
        block_size = (6,6)
        cols = []
        rowp = []
        nelem_per_comp = self.tree.nelem_per_comp

        # immediately ignore bcs from sparsity
        bcs = [_ for _ in range(6)]
        self.bcs = bcs
        self.keep_dof = [_ for _ in range(self.ndof) if not(_ in bcs)]

        # make a cols dict that says for each row : which cols it has
        cols_dict = {inode : [] for inode in range(1, self.nnodes)} # skip zeroth node because it's constrained
        for icomp in range(self.ncomp):
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]
                # print(f"{elem_nodes=}")
                node1 = elem_nodes[0]; node2 = elem_nodes[1]
                if node1 == 0 or node2 == 0: # nodal bcs = [0]
                    continue
                cols_dict[node1] += elem_nodes
                cols_dict[node2] += elem_nodes

        # now convert cols dict into a rowp, cols CSR pattern
        rowp += [0] # start with 0
        nnzb = 0
        for brow in range(self.nnodes):
            if brow == 0: continue # skip this constrained node
            c_cols = np.sort(np.unique(cols_dict[brow]))
            for col in c_cols:
                nnzb += 1
                cols += [col-1] # one less since we removed col 0
            rowp += [nnzb]

        # make np.arrays of rowp, cols, data
        rowp = np.array(rowp)
        cols = np.array(cols)         
        data = np.zeros((len(cols), *block_size))
        data2 = np.zeros((len(cols), *block_size))

        # print(f"{rowp=}\n{cols=}\n{data.shape=}")

        # make Kmat and Mmat initial nonzero patterns (with all zeros), reduced matrices with bcs gone
        self.Kr = bsr_matrix((data, cols, rowp), shape=matrix_shape, blocksize=block_size)
        self.Mr = bsr_matrix((data2, cols, rowp), shape=matrix_shape, blocksize=block_size)

    def _build_sparse_matrices(self, x):

        lengths = x[0::3]
        self.xpts = self.tree.get_xpts(lengths)
        nelem_per_comp = self.tree.nelem_per_comp

        rowp = self.Kr.indptr
        cols = self.Kr.indices

        def add_to_bsr(mat, elem_mat, row_block, col_block):
            for i in range(self.nnodes-1):
                for jp in range(rowp[i], rowp[i+1]):
                    j = cols[jp]
                    if i == row_block and j == col_block:
                        mat.data[jp, :, :] += elem_mat

        # zero out Kmat and Mmat again
        self.Kr.data *= 0.0
        self.Mr.data *= 0.0

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
            ref_axis = np.zeros((3,))
            ref_axis[rem_orient_ind[0]] = 1.0

            # set element xpts (for one element in straight comp, all same Kelem + Melem then)
            elem_xpts = np.concatenate([xpt1, xpt2], axis=0)
            # switch to xpts in x-dir then permute x,y,z
            # elem_xpts = np.array([0.0] * 3 + [L/nelem_per_comp, 0.0, 0.0])
            qvars = np.array([0.0]*12)

            # get the Kelem and Melem for this component
            Cfull = get_constitutive_data(self.material, t1, t2)
            CK = Cfull[:6]
            CM = Cfull[6:]
            Kelem = get_stiffness_matrix(elem_xpts, qvars, ref_axis, CK)
            Melem = get_mass_matrix(elem_xpts, qvars, ref_axis, CM)

            # now do assembly step for each element
            start = nelem_per_comp * icomp
            for ielem in range(start, start + nelem_per_comp):
                elem_nodes = self.elem_conn[ielem]

                # add all four blocks of Kelem to the matrix and same for Melem
                for i,inode in enumerate(elem_nodes):
                    if inode == 0: continue
                    for j,jnode in enumerate(elem_nodes):
                        if jnode == 0: continue # one less node bc node 0 constrained
                        sub_Kelem = Kelem[6*i:6*(i+1),6*j:6*(j+1)]
                        add_to_bsr(self.Kr, sub_Kelem, inode-1, jnode-1)
                        sub_Melem = Melem[6*i:6*(i+1),6*j:6*(j+1)]
                        add_to_bsr(self.Mr, sub_Melem, inode-1, jnode-1)

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