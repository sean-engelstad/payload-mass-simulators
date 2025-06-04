import sys, numpy as np, os, time
from payload_mass_sim import *
import argparse
from _utils import test

# INPUT OPT DESIGN HERE
# ---------------------

opt_design = np.array([3.85955282e-01, 1.57470312e-01, 4.85090707e-02, 7.13300625e-02,
       1.79486867e-02, 6.99841007e+00, 4.36679376e-01, 1.39146319e+00,
       5.04779082e-02, 5.88033481e-02, 2.30019816e-01, 9.25088124e-02,
       8.77775743e+00, 2.00436130e-01, 2.32590104e-01, 5.93937518e-02,
       2.71309049e-02, 3.42781257e-02, 3.07150617e-02, 3.99069649e-01,
       5.20415342e-01, 4.51909170e-01, 6.87396183e-02, 5.11803600e-02,
       1.05027105e-01, 6.89447664e-02, 7.30376182e+00, 3.59458233e-01,
       1.56706915e+00, 1.30913065e-01, 9.85800779e-02, 6.77662952e-02,
       5.56507379e-02, 4.93124840e+00, 1.31019959e-01, 2.02658237e-01,
       3.05668847e-01, 4.93504212e-02, 4.01843739e-01, 4.53400093e-02,
       7.92086023e+00, 1.99402082e-01, 7.67197043e-02, 4.83303456e-02,
       7.24453467e-03, 3.72869731e-02, 6.90511142e-03, 1.61833268e-01,
       1.58710371e-01, 1.41025059e-01, 3.39176662e-01, 2.57961504e-02,
       3.54567651e-01, 2.58444116e-02, 3.31001211e+00, 1.45689123e-01,
       2.39599846e-01, 1.57146366e-01, 1.08516082e-02, 1.51282760e-01,
       1.08772308e-02, 1.02547711e+00, 1.69489627e-01])

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

    cg = tree.get_centroid(opt_design)
    print(f"{cg=}")
    exit()
    # tree.centroid_FD_test(init_design, h=1e-5, idv='all')
    # exit()

    beam3d.get_frequencies(opt_design)
    beam3d.write_freq_to_vtk(nmodes=4, file_prefix=f"{args.output}/")

    beam3d.solve_static(opt_design)
    failure = beam3d.get_failure_index(opt_design)
    print(f"{failure=}")
    beam3d.write_static_to_vtk(file_prefix=f"{args.output}/")