import sys, numpy as np, os, time
from payload_mass_sim import *

if not os.path.exists("_modal"):
    os.mkdir("_modal")

if __name__ == "__main__":

    material = Material.aluminum()

    def loc_dv_list(length, thick=5e-3):
        # starting dv list
        return ([length] + [thick]*4 + [0.0, 0.5])

    # tree data structure
    # -------------------
    # directions [0,1,2,3,4,5] equiv to x-,x+,y-,y+,z-,z+

    # base and level 1
    init_design = loc_dv_list(0.3) * 9
    tree_start_nodes = [0] + [1]*4 + [2,3,4,5]
    tree_directions = [5] + [0,1,2,3] + [5]*4

    Llittle = 0.2 # 0.1

    # level 2, x- branch
    init_design += loc_dv_list(Llittle) * 8
    tree_start_nodes += [6]*4 + [10,11,12,13]
    tree_directions += [0,1,2,3] + [5]*4

    # level 2, x+ branch
    init_design += loc_dv_list(Llittle) * 8
    tree_start_nodes += [7]*4 + [18,19,20,21]
    tree_directions += [0,1,2,3] + [5]*4

    # level 2, y- branch
    init_design += loc_dv_list(Llittle) * 8
    tree_start_nodes += [8]*4 + [26,27,28,29]
    tree_directions += [0,1,2,3] + [5]*4

    # level 2, y+ branch
    init_design += loc_dv_list(Llittle) * 8
    tree_start_nodes += [9]*4 + [34,35,36,37]
    tree_directions += [0,1,2,3] + [5]*4

    tree = TreeData(
        tree_start_nodes=tree_start_nodes,
        tree_directions=tree_directions,
        nelem_per_comp=5 
    )
    init_design = np.array(init_design)
    ncomp = tree.ncomp
    num_dvs = init_design.shape[0]
    inertial_data = InertialData([1, 0, 0])

    beam3d = BeamAssemblerAdvanced(material, tree, inertial_data)

    demo = True
    if demo:
        # now build 3D beam solver and solve the eigenvalue problem
        nmodes = 15
        freqs = beam3d.get_frequencies(init_design, nmodes)
        print(f"{freqs=}")
        beam3d.write_freq_to_vtk(nmodes=nmodes, file_prefix="_modal/")
        # beam3d.get_frequency_gradient(init_design, 0)

    # beam3d.plot_eigenmodes(
    #     nmodes=2,
    #     show=True,
    #     def_scale=0.5
    # )

    debug = False
    if debug:
        # FD test on the gradients
        for imode in range(4):
            beam3d.freq_FD_test(init_design, imode, h=1e-3)
            beam3d.dKdx_FD_test(init_design, imode, h=1e-5)
            beam3d.dMdx_FD_test(init_design, imode, h=1e-5)