
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys, numpy as np, os, time
from payload_mass_sim import *

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
ncomp = 9
init_design = init_design = np.array([0.3, 5e-3, 5e-3]*ncomp)

# optimal design - V1 (first 4 freq match)
# opt_design = [0.08125551, 0.02211063, 0.07895405, 0.14713621, 0.01711272,
#        0.05670854, 0.14552369, 0.00420714, 0.02302042, 0.17597928,
#        0.04472327, 0.0450355 , 0.11794632, 0.05531567, 0.00601055,
#        0.08715245, 0.00400008, 0.00345651, 0.14678793, 0.0039524 ,
#        0.00351281, 0.12807156, 0.00403881, 0.00427038, 0.16456358,
#        0.00351356, 0.00346978]

# optimal design - V2 (first 4 freq match + mass match
opt_design = np.array([0.51557176, 0.22839125, 0.20479483, 0.39822018, 0.08301871,
       0.02212506, 0.64312029, 0.11802109, 0.16108851, 0.7288425 ,
       0.19153433, 0.13970275, 0.8891137 , 0.13101526, 0.11585078,
       0.2908164 , 0.03151235, 0.01452752, 0.28787797, 0.03079086,
       0.01134946, 0.29856402, 0.01182893, 0.02284578, 0.20969153,
       0.00601115, 0.14087715])



tree = TreeData(
    tree_start_nodes=[0] + [1]*4 + [2,3,4,5],
    tree_directions=[5] + [0,1,2,3] + [5]*4,
    nelem_per_comp=10 # 1 since we only want to plot the beams (no FEM mesh here, just visualization)
)
material = Material.aluminum()
inertial_data = InertialData([1.0, 0.0, 0.0])

beam3d = BeamAssembler(material, tree, inertial_data, rho_KS=10.0, safety_factor=1.5)


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
plt.savefig("_modal/init-design.png")
plt.close('all')

# Create figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_design(fig, ax, opt_design, origin=[0.0]*3, show=False)
plt.savefig("_modal/opt-design.png")

freqs = beam3d.get_frequencies(opt_design)

beam3d.write_freq_to_vtk(nmodes=5, file_prefix="_modal/")

# debug linear static solve
beam3d.solve_static(opt_design)
fail_index = beam3d.get_failure_index(opt_design)
print(f"{fail_index=}")
beam3d.write_static_to_vtk(file_prefix="_modal/")