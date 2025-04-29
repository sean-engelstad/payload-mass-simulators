import numpy as np

def get_node_pt(ibasis):
    return 

def basis_fcn(ind, xi):
    if ind == 0:
        return 0.5 * (1 - xi)
    else:
        return 0.5 * (1 + xi)

def get_beam_node_normals(N_BASIS, xpts, axis):
    # get beam normal directions
    # ref axis is basically the second direction

    fn1 = np.zeros((3*N_BASIS,))
    fn2 = np.zeros((3*N_BASIS,))

    for ibasis in range(N_BASIS):
        pt = -1.0 + 2.0 * ibasis # -1 or 1

        # get fields grad X0xi
        X0xi = xpts[0:3] * -0.5 + xpts[3:6] * 0.5
        t1 = X0xi / np.linalg.norm(X0xi)

        # get second direction
        t2 = axis - np.dot(t1, axis) * t1
        t2 /= np.linalg.norm(t2)

        # get remaining direction
        t3 = np.cross(t1, t2)

        # store node normals
        fn1[3*ibasis:3*(ibasis+1)] = t2[:]
        fn2[3*ibasis:3*(ibasis+1)] = t3[:]
        
    return fn1, fn2

def compute_director(qvars, fn):
    d = np.zeros((6,), dtype=np.complex128)
    # uses rotational DOF here so last 3 of each node
    d[:3] = np.cross(qvars[3:6], fn[:3])
    d[3:] = np.cross(qvars[9:12], fn[3:])
    return d

def compute_tying_strains(xpts, fn1, fn2, qvars, d1, d2):
    # only considers the transverse shear stresses here (2 strains)
    ety = np.zeros((2,), dtype=np.complex128)
    # xi = 1.0 # weird that it's not just 0 then
    # would just interp to the second point Xxi, Uxi etc.
    # double check this?

    # prelim
    Xxi = 0.5 * (xpts[3:] - xpts[:3])
    Udisp_xi = 0.5 * (qvars[6:9] - qvars[0:3]) # only u,v,w considered here not rotations

    # first the g12 strain (linear model)
    xi = 0.0 # which means basis functions are just 0.5 each
    d0 = 0.5 * (d1[:3] + d1[3:])
    n0 = 0.5 * (fn1[:3] + fn1[3:])
    ety[0] = 0.5 * (np.dot(Xxi, d0) + np.dot(n0, Udisp_xi))
    
    # then the g13 strain (linear model)
    d0 = 0.5 * (d2[:3] + d2[3:])
    n0 = 0.5 * (fn2[:3] + fn2[3:])
    ety[1] = 0.5 * (np.dot(Xxi, d0) + np.dot(n0, Udisp_xi))

    return ety