import numpy as np
import pyvista as pv

def write_beam_modes_to_vtk(filename, node_coords, elements, mode_shapes=None, thicknesses=None, nonstruct_masses=None):
    """
    Write a beam structure with beam elements to a VTK file.

    Parameters:
    -----------
    filename : str
        Path to the output VTK file (.vtp or .vtk).
    node_coords : np.ndarray of shape (N, 3)
        Coordinates of the beam nodes.
    elements : list of tuples or np.ndarray of shape (M, 2)
        Each entry is a 2-node beam element (start_node_id, end_node_id).
    mode_shapes : np.ndarray of shape (N, 6), optional
        Mode shape vector at each node: 3 translations + 3 rotations.
    """
    points = node_coords
    lines = []
    for elem in elements:
        lines.append(2)
        lines.extend(elem)

    poly = pv.PolyData()
    poly.points = points
    poly.lines = np.array(lines)

    if mode_shapes is not None:
        if mode_shapes.shape[1] != 6:
            raise ValueError("mode_shapes must be of shape (N, 6)")
        poly.point_data["mode_disp"] = mode_shapes[:, 0:3]
        poly.point_data["mode_rot"] = mode_shapes[:, 3:6]

    if thicknesses is not None:
        if thicknesses.shape[1] != 2:
            raise ValueError("thicknesses must be of shape (N, 2)")
        poly.point_data["t1"] = thicknesses[:, 0]
        poly.point_data["t2"] = thicknesses[:, 1]

    if nonstruct_masses is not None:
        poly.point_data["M"] = nonstruct_masses

    poly.save(filename)
    # print(f"Saved beam structure with mode shapes to {filename}")

def write_beam_static_to_vtk(filename, node_coords, elements, disps, stresses=None, vm_stress=None, thicknesses=None, nonstruct_masses=None):
    """
    Write a static beam solution to a VTK file.

    Parameters:
    -----------
    filename : str
        Path to the output VTK file (.vtp or .vtk).
    node_coords : np.ndarray of shape (N, 3)
        Coordinates of the beam nodes.
    elements : list of tuples or np.ndarray of shape (M, 2)
        Each entry is a 2-node beam element (start_node_id, end_node_id).
    disps : np.ndarray of shape (N, 6)
        Displacement vector at each node: 3 translations + 3 rotations.
    stresses : np.ndarray of shape (N, 6), optional
        Stress tensor components at each node: (σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx) (actually not that, see code, fix later).
    vm_stress : np.ndarray of shape (N,), optional
        Scalar von Mises stress at each node.
    """
    points = node_coords
    lines = []
    for elem in elements:
        lines.append(2)
        lines.extend(elem)

    poly = pv.PolyData()
    poly.points = points
    poly.lines = np.array(lines)

    if disps.shape[1] != 6:
        raise ValueError("disps must be of shape (N, 6)")

    poly.point_data["disp"] = disps[:, 0:3]
    poly.point_data["rot"] = disps[:, 3:6]

    if stresses is not None:
        if stresses.shape[1] != 6:
            raise ValueError("stresses must be of shape (N, 6)")
        # Store stress components individually or as a tensor
        poly.point_data["stress"] = stresses  # [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx]

    if vm_stress is not None:
        if vm_stress.ndim != 1 or vm_stress.shape[0] != node_coords.shape[0]:
            raise ValueError("vm_stress must be of shape (N,)")
        poly.point_data["von_mises"] = vm_stress

    if thicknesses is not None:
        if thicknesses.shape[1] != 2:
            raise ValueError("thicknesses must be of shape (N, 2)")
        poly.point_data["t1"] = thicknesses[:, 0]
        poly.point_data["t2"] = thicknesses[:, 1]

    if nonstruct_masses is not None:
        poly.point_data["M"] = nonstruct_masses

    poly.save(filename)
    # print(f"Saved static beam solution to {filename}")
