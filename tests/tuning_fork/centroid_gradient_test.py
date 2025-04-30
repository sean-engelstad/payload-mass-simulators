import sys, numpy as np, os, time
from payload_mass_sim import *

x = np.array([2,1,1] * 9)
# x = np.array([2,1,1])

# tree = TreeData(
#     tree_start_nodes=[0] + [1],
#     tree_directions=[5] + [0],
#     # tree_start_nodes=[0],
#     # tree_directions=[5],
#     nelem_per_comp=10,
# )

tree = TreeData(
        tree_start_nodes=[0] + [1]*4 + [2,3,4,5],
        tree_directions=[5] + [0,1,2,3] + [5]*4,
        nelem_per_comp=10
    )

material = Material.aluminum()
beam3d = Beam3DTree(material, tree)

# centroid = tree.get_centroid(x)
# print(f"{centroid=}")
# tree.centroid_FD_test(x)

for imode in range(4):
    beam3d.freq_FD_test(x, imode, h=1e-3)
    beam3d.dKdx_FD_test(x, imode, h=1e-5)
    beam3d.dMdx_FD_test(x, imode, h=1e-5)