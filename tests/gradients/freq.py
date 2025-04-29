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
    
    init_design = np.array([2.0, 7e-2, 2e-2])
    beam3d = BeamAssembler(material, tree)
    x = init_design

    # now test the frequencies
    h = 1e-5
    for imode in range(5):
    # for imode in range(1):
        # beam3d.dKdx_FD_test(x, imode, h=h)
        # beam3d.dMdx_FD_test(x, imode, h=h)
        beam3d.freq_FD_test(init_design, imode, h=h)
