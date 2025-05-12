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
        tree_start_nodes=[0] + [1]*4 + [2,3,4,5],
        tree_directions=[5] + [0,1,2,3] + [5]*4,
        nelem_per_comp=10 #10
    )

    inertial_data = InertialData([1.0, 0.0, 0.0])

    # initial design variables
    ncomp = tree.ncomp
    t1 = t2 = 5e-3 # m
    L = 1.0 # m
    init_design = np.array([L, t1, t2]*ncomp)
    num_dvs = init_design.shape[0]

    # should lead to:
    # A = 1e-2, Iy = Iz = 8.333e-6, Iyz = 0
    # J = 2*Iy=1.666e-5, all other properties zero
    beam3d = BeamAssembler(material, tree, inertial_data, 
        rho_KS=10.0, safety_factor=1.5)

    # now build 3D beam solver and solve the eigenvalue problem
    freqs = beam3d.get_frequencies(init_design)
    print(f"{freqs=}")
    
    nmodes = 5
    # beam3d.plot_eigenmodes(
    #     nmodes=nmodes,
    #     show=True,
    #     def_scale=3.0
    # )
    beam3d.write_freq_to_vtk(nmodes=5, file_prefix="_modal/")

    # debug linear static solve
    beam3d.solve_static(init_design)
    fail_index = beam3d.get_failure_index(init_design)
    print(f"{fail_index=}")
    beam3d.write_static_to_vtk(file_prefix="_modal/")