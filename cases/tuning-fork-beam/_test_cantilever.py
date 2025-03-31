import sys, numpy as np, os, time
sys.path.append("beam-fea")
from solver_3D import *
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
    init_design = np.array([0.3, 5e-3, 1e0]*ncomp)
    num_dvs = init_design.shape[0]

    beam3d = Beam3DTree(material, tree)

    rhoA = material.rho * 1e0 * 5e-3
    EI = material.E * 1e0 * (5e-3)**3 / 12.0
    omega_hat = np.sqrt(EI / rhoA)
    L = 0.3
    nat_freq = (np.array([0.596864, 1.49418]) * np.pi / L / 2.0)**2 * omega_hat
    print(f"{nat_freq=}")

    # now build 3D beam solver and solve the eigenvalue problem
    freqs = beam3d.get_frequencies(init_design)
    print(f"{freqs=}")
    nmodes = 5
    beam3d.plot_eigenmodes(
        nmodes=nmodes,
        show=True,
        def_scale=0.1
    )

    
