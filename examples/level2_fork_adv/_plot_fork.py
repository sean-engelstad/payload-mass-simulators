import sys, numpy as np, os, time
from payload_mass_sim import *
import argparse
from _utils import test

# INPUT OPT DESIGN HERE
# ---------------------

opt_design = np.array([3.74609217e-02, 3.64899063e-02, 1.33015254e-02, 5.78203469e-02,
       1.32351143e-02, 2.87198500e+01, 1.00000000e-01, 3.40027699e-01,
       5.06188813e-02, 5.27766688e-02, 8.73305881e-02, 7.22475702e-02,
       2.56445912e-01, 1.74180543e-01, 2.86260881e-01, 2.56984405e-01,
       2.71941592e-01, 7.05678552e-02, 1.51212874e-01, 3.37645591e-01,
       1.52364272e-01, 6.82921417e-02, 6.04718197e-02, 9.92367078e-02,
       5.79164312e-02, 1.10444057e-01, 7.60622379e-01, 8.31611069e-01,
       2.97496377e-02, 5.11093912e-02, 6.39422267e-02, 3.86220291e-02,
       6.82035859e-02, 4.72295184e-01, 7.98072062e-01, 3.95077058e-01,
       1.32606374e-02, 9.89954083e-03, 2.28335876e-02, 5.23128287e-02,
       1.13317284e-01, 3.87391071e-01, 6.16957402e-01, 6.61532918e-02,
       3.68821977e-02, 6.48321816e-02, 4.05651737e-02, 1.58695373e-01,
       1.00000000e-01, 1.40814376e+00, 1.15507084e-01, 9.71654704e-02,
       1.32103827e-01, 1.56299842e-01, 1.38378240e-01, 1.00000000e-01,
       9.45178152e-01, 1.57756783e-01, 1.35352847e-01, 1.72811965e-01,
       1.14040159e-01, 1.45630112e-01, 1.00000000e-01])

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

    beam3d.get_frequencies(opt_design)
    beam3d.write_freq_to_vtk(nmodes=4, file_prefix=f"{args.output}/")

    beam3d.solve_static(opt_design)
    failure = beam3d.get_failure_index(opt_design)
    print(f"{failure=}")
    beam3d.write_static_to_vtk(file_prefix=f"{args.output}/")