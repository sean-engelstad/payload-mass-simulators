import numpy as np
from payload_mass_sim import *

# xpts = np.array([0,0,0] + [0,0,1])
# axis = np.array([0,1,0])

xpts = np.array([0,0,0] + [0,0,1])
axis = np.array([1,0,0])

fn1, fn2 = get_beam_node_normals(2, xpts, axis)
print(f"{fn1=} {fn2=}")