import sys, numpy as np, os, time
from payload_mass_sim import *

if not os.path.exists("_modal"):
    os.mkdir("_modal")

if __name__ == "__main__":

    material = Material.aluminum()

    # test out the tree data structure
    # TODO : modify tree start nodes and directions graph
    #    in order to have level 2 optimization
    tree = TreeData(
        tree_start_nodes=[0],
        tree_directions=[0],
        nelem_per_comp=10
    )
    # not sensitive to the number of elements

    # initial design variables
    ncomp = tree.ncomp
    # t1 = t2 = 5e-3 # m
    t1 = t2 = 1e-2
    L = 1.0 # m
    init_design = np.array([L, t1, t2]*ncomp)
    num_dvs = init_design.shape[0]

    # should lead to:
    # A = 1e-2, Iy = Iz = 8.333e-6, Iyz = 0
    # J = 2*Iy=1.666e-5, all other properties zero

    beam3d = Beam3DTree(material, tree)

    # debug make other frequencies really high to just use bending 1
    # beam3d._axial_mult = 1e4
    # beam3d._tor_mult = 1e4
    # beam3d._bend1_mult = 1.0
    # beam3d._bend2_mult = 1e4

    # debug to make all but bending low eigenvalues
    # beam3d._A = 1e4
    # beam3d._J = 1e4

    # now build 3D beam solver and solve the eigenvalue problem
    freqs = beam3d.get_frequencies(init_design)
    print(f"{freqs=}")
    nmodes = 5
    beam3d.plot_eigenmodes(
        nmodes=nmodes,
        show=False,
        def_scale=3.0
    )