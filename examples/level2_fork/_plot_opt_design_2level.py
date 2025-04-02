
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys, numpy as np, os, time
sys.path.append("beam-fea")
from solver_3D import *

def draw_box(ax, center, size, color='cyan', alpha=0.3):
    """
    Draws a 3D rectangular box centered at `center` with dimensions `size`.
    """
    x, y, z = center
    dx, dy, dz = size

    # Define the 8 corner points of the box
    corners = np.array([
        [x - dx/2, y - dy/2, z - dz/2],
        [x + dx/2, y - dy/2, z - dz/2],
        [x + dx/2, y + dy/2, z - dz/2],
        [x - dx/2, y + dy/2, z - dz/2],
        [x - dx/2, y - dy/2, z + dz/2],
        [x + dx/2, y - dy/2, z + dz/2],
        [x + dx/2, y + dy/2, z + dz/2],
        [x - dx/2, y + dy/2, z + dz/2]
    ])

    # Define the six faces as lists of corners
    faces = [
        [corners[j] for j in [0, 1, 2, 3]],  # Bottom face
        [corners[j] for j in [4, 5, 6, 7]],  # Top face
        [corners[j] for j in [0, 1, 5, 4]],  # Side 1
        [corners[j] for j in [2, 3, 7, 6]],  # Side 2
        [corners[j] for j in [0, 3, 7, 4]],  # Side 3
        [corners[j] for j in [1, 2, 6, 5]],  # Side 4
    ]

    # Add the collection to the axis
    ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=alpha, edgecolor='k'))

    return corners  # Return corners for axis scaling

# init design
ncomp = 25
init_design = init_design = np.array([0.3, 5e-3, 5e-3]*ncomp)

# optimal design
opt_design = [0.51356357, 0.58936876, 0.59449344, 0.54337107, 0.66756475,
       0.66879658, 0.54139735, 0.2504392 , 0.24274076, 0.06250699,
       0.79819811, 0.7973222 , 0.28402404, 0.98954866, 0.97708781,
       0.89280352, 0.5475172 , 0.56755643, 0.09663544, 0.52104368,
       0.52605036, 0.18319159, 0.64093505, 0.61873378, 0.17402498,
       0.92646682, 0.95629892, 0.21413584, 0.29807207, 0.29891428,
       0.19414718, 0.18142242, 0.18275822, 1.22820854, 0.21288764,
       0.33856259, 0.68982044, 0.93453992, 0.07559592, 0.01568144,
       0.20018254, 0.17895316, 0.23159184, 0.21644126, 0.21785645,
       0.34368306, 0.28154457, 0.50129818, 0.98122663, 0.54265563,
       0.64383522, 0.10464439, 0.08448421, 0.03222893, 0.33601293,
       0.03444808, 0.04266918, 0.24334277, 0.19528914, 0.0118297 ,
       0.05462177, 0.26193797, 0.14976567, 0.01092886, 0.04308376,
       0.05148861, 0.23054725, 0.04713601, 0.01445863, 0.12365468,
       0.229301  , 0.04899589, 0.19800888, 0.01182331, 0.18818442]

tree = TreeData(
    tree_start_nodes=[0] + [1]*4 + [2,3,4,5] + [6,7,8,9]*2 + [10,11,12,13,14,15,16,17],
    tree_directions=[5] + [0,1,2,3] + [5]*4 + [3,3,1,1] + [2,2,0,0] + [5]*8,
    nelem_per_comp=10 
)

init_lengths = [1.0]*9 + [0.5] * 16
xpts = tree.get_xpts(init_lengths)
tree.plot_mesh(xpts)

def plot_design(fig, ax, x_design, origin, show=True):

    # beams = [
    #     ((0, 0, 0), (2, 1, 5)),  # Center at (0,0,0), size (2x1x5)
    #     ((3, 1, 2), (1.5, 1.5, 4)),  # Another beam shifted in space
    #     ((-2, -1, 3), (1, 2, 6)),  # Another beam
    # ]

    lengths = x_design[0::3]
    tree_xpts = tree.get_xpts(lengths, origin)
    # tree_start_nodes = 
    # print(f"{tree_xpts=}")

    # make list of each beam and its boundaries
    beams = []
    for icomp in range(9):
        L = x_design[3*icomp]
        t1 = x_design[3*icomp+1]
        t2 = x_design[3*icomp+2]

        start_node = tree.tree_start_nodes[icomp]
            
        start_pt = tree_xpts[3*start_node:3*start_node+3]
        end_pt = tree_xpts[3*icomp+3:3*icomp+6]

        _center = 0.5 * (start_pt + end_pt)
        center = (_center[0], _center[1], _center[2])

        dx = end_pt - start_pt
        if dx[0] != 0:
            size = (L, t1, t2)
        elif dx[1] != 0:
            size = (t1, L, t2)
        else:
            size = (t1, t2, L)

        beams += [(center, size)]
    print(f"{beams=}")

    # Collect all corners to adjust limits
    all_corners = []
    ct = 0
    jet_colors = plt.cm.jet(np.linspace(0.0, 1.0, 9))
    for center, size in beams:
        corners = draw_box(ax, center, size, color=jet_colors[ct])
        all_corners.extend(corners)
        ct += 1

    # Convert to array for easier min/max calculation
    all_corners = np.array(all_corners)

    # Get global min and max for each axis
    x_min, y_min, z_min = np.min(all_corners, axis=0)
    x_max, y_max, z_max = np.max(all_corners, axis=0)

    # Set equal aspect ratio with proper limits
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x, mid_y, mid_z = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0, (z_max + z_min) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])  # Equal scaling for x, y, z

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if show:
        plt.show()


# save initial design
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plot_design(fig, ax, init_design, origin=[0.0]*3, show=False)
plt.savefig("_modal_2level/init-design.png")
plt.close('all')

# Create figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_design(fig, ax, opt_design, origin=[0.0]*3, show=False)
plt.savefig("_modal_2level/opt-design.png")
