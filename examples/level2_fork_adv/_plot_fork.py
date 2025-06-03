import sys, numpy as np, os, time
from payload_mass_sim import *
import argparse
from _utils import test

# INPUT OPT DESIGN HERE
# ---------------------

opt_design = np.array([0.20224865, 0.19090424, 0.02962008, 0.08673134, 0.01777555,
       0.1       , 0.1031568 , 1.62026392, 0.07564767, 0.02570425,
       0.42298174, 0.01163898, 0.1       , 0.15312075, 0.27795549,
       0.02086831, 0.02113705, 0.02933038, 0.00993898, 0.1       ,
       0.14241961, 2.18674332, 0.12804176, 0.0937638 , 0.15515426,
       0.04521069, 0.1       , 0.33433394, 1.84795265, 0.11528713,
       0.06107955, 0.10992842, 0.07607901, 0.1       , 0.55669942,
       0.75589684, 0.02606629, 0.04224058, 0.04928818, 0.01517419,
       0.1       , 0.11357341, 0.02078715, 0.01821216, 0.25953526,
       0.00341078, 0.05043633, 0.1       , 0.709542  , 0.33447894,
       0.06307702, 0.05378603, 0.06104064, 0.05281833, 0.1       ,
       0.1       , 0.08768695, 0.11145058, 0.0192993 , 0.12491264,
       0.02094528, 0.1       , 0.1       ])

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

    beam3d = BeamAssemblerAdvanced(material, tree, inertial_data, rho_KS=3.0)

    # plot eigenmodes, static analysis, etc.
    # --------------------------------------

    beam3d.get_frequencies(opt_design)
    beam3d.write_freq_to_vtk(nmodes=4, file_prefix=f"{args.output}/")

    beam3d.solve_static(opt_design)
    failure = beam3d.get_failure_index(opt_design)
    print(f"{failure=}")
    beam3d.write_static_to_vtk(file_prefix=f"{args.output}/")