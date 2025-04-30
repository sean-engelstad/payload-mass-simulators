import sys, numpy as np, os, time
from payload_mass_sim import *
from pyoptsparse import SNOPT, Optimization

if __name__ == "__main__":

    material = Material.aluminum()

    # test out the tree data structure
    # TODO : modify tree start nodes and directions graph
    #    in order to have level 2 optimization
    tree = TreeData(
        tree_start_nodes=[0],
        tree_directions=[1],
        nelem_per_comp=10
    )
    # init_lengths = [1.0]*tree.ncomp
    # xpts = tree.get_xpts(init_lengths)
    # tree.plot_mesh(xpts)

    # initial design variables
    ncomp = tree.ncomp
    init_design = np.array([1.0, 1e-2, 1e-3]*ncomp)
    # init_design = np.array([1.0, 1e-2, 1e-2]*ncomp)
    num_dvs = init_design.shape[0]

    beam3d = BeamAssembler(material, tree)

    # analytic soln ------------------
    E, G, ks, rho = material.E, material.G, material.k_s, material.rho
    L, t1, t2 = init_design[0], init_design[1], init_design[2]
    A, Iz = t1 * t2, t2**3 * t1 / 12.0
    omega_EB = np.array([1.875, 4.694, 7.855])**2 * np.sqrt(E * Iz / rho / A / L**4)
    TS_denom = 1.0 + np.array([1,2,3])**2 * np.pi**2 * E * t1**2 / ks / G / L**2
    omega_TS = omega_EB / np.sqrt(TS_denom)
    print(f"{omega_EB=}\n{omega_TS=}")

    # now build 3D beam solver and solve the eigenvalue problem
    freqs = beam3d.get_frequencies(init_design)
    print(f"{freqs=}")
    nmodes = 5
    beam3d.plot_eigenmodes(
        nmodes=nmodes,
        show=False,
        def_scale=0.1
    )

    
