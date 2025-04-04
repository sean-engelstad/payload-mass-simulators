import sys, numpy as np, os, time
sys.path.append("beam-fea")
from solver_3D import *

# x = np.array([2,1,1] + [2,1,1])
x = np.array([2,1,1])

tree = TreeData(
    # tree_start_nodes=[0] + [1],
    # tree_directions=[5] + [0],
    tree_start_nodes=[0],
    tree_directions=[5],
    nelem_per_comp=10,
)

centroid = tree.get_centroid(x)
print(f"{centroid=}")

tree.centroid_FD_test(x)