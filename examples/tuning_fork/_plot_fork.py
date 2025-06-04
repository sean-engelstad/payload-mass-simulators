import sys, numpy as np, os, time
from payload_mass_sim import *
import argparse
from _utils import test

# INPUT OPT DESIGN HERE
# ---------------------

opt_design = np.array([0.40774859, 0.08255792, 0.03415702, 0.06825633, 0.04095295,
       9.28898016, 0.38175169, 1.45872071, 0.04118441, 0.06659624,
       0.0673919 , 0.07012606, 0.51104885, 0.55269104, 1.88120946,
       0.04415318, 0.05430111, 0.07499303, 0.05493083, 0.1       ,
       0.34653864, 1.495371  , 0.03223959, 0.04089094, 0.08066521,
       0.04987093, 1.26752227, 0.36283395, 1.46221323, 0.0677828 ,
       0.03856303, 0.06760171, 0.04806182, 1.55538026, 0.47130026,
       0.31295402, 0.05710853, 0.04836035, 0.05769831, 0.04973134,
       1.60053898, 0.34845109, 0.3010551 , 0.03755707, 0.03377524,
       0.03746596, 0.03375339, 0.1       , 0.36433832, 0.28032699,
       0.04016259, 0.03705806, 0.04063407, 0.03718169, 1.33976544,
       0.35550124, 0.22120081, 0.03794393, 0.03511473, 0.03811748,
       0.03514913, 0.33701898, 0.37835918])

# ----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Constraint toggles")
    parser.add_argument('--level', type=int, default=1, help='Level of tuning fork (only goes up to 2 at the moment)')
    parser.add_argument('--output', type=str, default="level1", help="output directory")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    material = Material.aluminum()

    def loc_dv_list(length, thick=5e-3):
        # starting dv list
        return ([length] + [thick]*4 + [0.1, 0.5])

    # tree data structure
    # -------------------
    # directions [0,1,2,3,4,5] equiv to x-,x+,y-,y+,z-,z+

    # base and level 1
    init_design = loc_dv_list(0.3, 3e-2) * 9
    tree_start_nodes = [0] + [1]*4 + [2,3,4,5]
    tree_directions = [5] + [0,1,2,3] + [5]*4

    # init_design = loc_dv_list(0.3, 3e-2) * 2
    # tree_start_nodes = [0] + [1]
    # tree_directions = [5] + [0]

    Llittle = 0.2 # 0.1

    # include_level2 = True
    include_level2 = False # just level1

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

    ncomp = tree.ncomp
    num_dvs = init_design.shape[0]
    inertial_data = InertialData([1, 0, 0])

    beam3d = BeamAssemblerAdvanced(material, tree, inertial_data, rho_KS=10.0)

    # plot eigenmodes, static analysis, etc.
    # --------------------------------------

    # cg = tree.get_centroid(opt_design)
    # print(f"{cg=}")
    # exit()
    # tree.centroid_FD_test(init_design, h=1e-5, idv='all')
    # exit()

    beam3d.get_frequencies(opt_design)
    beam3d.write_freq_to_vtk(nmodes=4, file_prefix=f"{args.output}/")

    beam3d.solve_static(opt_design)
    failure = beam3d.get_failure_index(opt_design)
    print(f"{failure=}")
    beam3d.write_static_to_vtk(file_prefix=f"{args.output}/")