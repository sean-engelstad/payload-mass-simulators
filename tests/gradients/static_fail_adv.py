import sys, numpy as np, os, time
from payload_mass_sim import *

if not os.path.exists("_modal"):
    os.mkdir("_modal")

if __name__ == "__main__":

    material = Material.aluminum()
    tree = TreeData(
        tree_start_nodes=[0],
        tree_directions=[0],
        # nelem_per_comp=10
        nelem_per_comp=1
    )

    inertial_data = InertialData([1.0, 0.0, 0.0])

    init_design = np.array([2.0] + [7e-2]*4 + [0.1, 0.5])
    beam3d = BeamAssemblerAdvanced(material, tree, inertial_data, rho_KS=10.0)
    x = init_design

    # now test the frequencies
    h = 1e-6
    # beam3d.mass_FD_test(x,h)
    # beam3d.dKdx_FD_static_test(x, h)
    # beam3d.dfail_dx_FD_test(x, h)
    # beam3d.dfail_du_FD_test(x, h)
    beam3d.dRdx_inertial_FD_test(x, h)
    # beam3d.rho_KS = 3.0
    # beam3d.fail_index_FD_test(x, h)
