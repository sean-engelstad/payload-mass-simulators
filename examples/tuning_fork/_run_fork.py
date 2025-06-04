import sys, numpy as np, os, time
from payload_mass_sim import *

if not os.path.exists("_modal"):
    os.mkdir("_modal")

if __name__ == "__main__":

    material = Material.aluminum()

    def loc_dv_list(length, thick=5e-3):
        # starting dv list
        return ([length] + [thick]*4 + [0.05, 0.5])

    # tree data structure
    # -------------------
    # directions [0,1,2,3,4,5] equiv to x-,x+,y-,y+,z-,z+

    init_design = loc_dv_list(0.3) * 1
    tree_start_nodes = [0]
    tree_directions = [5] 

    include_level1 = True
    if include_level1:
        # base and level 1
        init_design += loc_dv_list(0.3) * 8
        tree_start_nodes += [1]*4 + [2,3,4,5]
        tree_directions += [0,1,2,3] + [5]*4

    Llittle = 0.2 # 0.1

    include_level2 = True
    if include_level2:
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

    # for derivatives testing, want to move somewhat away from equal thickness design prob
    # init_design += np.random.rand(init_design.shape[0]) * 1e-2

    ncomp = tree.ncomp
    num_dvs = init_design.shape[0]
    inertial_data = InertialData([1, 0, 0])

    beam3d = BeamAssemblerAdvanced(material, tree, inertial_data, rho_KS=0.1)

    demo = False
    if demo:
        # now build 3D beam solver and solve the eigenvalue problem
        nmodes = 5
        freqs = beam3d.get_frequencies(init_design, nmodes)
        print(f"{freqs=}")
        beam3d.write_freq_to_vtk(nmodes=nmodes, file_prefix="_modal/")
        # beam3d.get_frequency_gradient(init_design, 0)

        beam3d.solve_static(init_design)
        fail_index = beam3d.get_failure_index(init_design)
        print(f"{fail_index=}")
        beam3d.write_static_to_vtk(file_prefix="_modal/")

    # beam3d.plot_eigenmodes(
    #     nmodes=2,
    #     show=True,
    #     def_scale=0.5
    # )

    debug = True
    if debug:
        # FD test on the gradients, all pass in new advanced beam assembler
        # for imode in range(1): #(4):
            # var scalings are a bit weird here
        # beam3d.dKdx_FD_test(init_design, imode=0, h=1e-6, idv=0)
            # beam3d.dMdx_FD_test(init_design, imode, h=1e-6)
            # beam3d.freq_FD_test(init_design, imode, h=1e-6)

        # beam3d.mass_FD_test(init_design, h=1e-6)
        # NOTE : if DV not active failure mode, deriv near 0.. not wrong, just how it works
        # beam3d.dKdx_FD_static_test(init_design, h=1e-5, idv='all') #'all'
        # beam3d.dfail_dx_FD_test(init_design, h=1e-5, idv='all')
        beam3d.dfail_du_FD_test(init_design, h=1e-5)
        # beam3d.dRdx_inertial_FD_test(init_design, h=1e-6, idv='all')

        # beam3d.rho_KS = 0.1
        beam3d.fail_index_FD_test(init_design, h=1e-5, idv=0)
            