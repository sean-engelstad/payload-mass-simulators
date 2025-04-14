"""
based on TACS timoshenko beam element
    the TACSBeamElement.h file
    @author : Sean Engelstad
"""
import numpy as np
from ._TS_utilities import *

N_VARS = 6
N_QUAD = 4
N_BASIS = 2

def add_stiffness_residual(
    xpts:np.ndarray, # 2 * 3 = 6 entries
    qvars:np.ndarray, # 2 * 6 = 12 entries
    ref_axis:np.ndarray, # 3 entries
) -> np.ndarray: 

    res = np.zeros((N_VARS,))
    
    fn1, fn2 = get_beam_node_normals(N_BASIS, xpts, axis)

    d1 = compute_director(qvars, fn1)
    d2 = compute_director(qvars, fn2)

    ety = compute_tying_strains(xpts, fn1, fn2, qvars, d1, d2)

    for iquad in range(2):
        xi = -1.0 + 2.0 * iquad

        u0xi = 0.5 * (qvars[6:] - qvars[:6])
        d01 = d1[:3] * basis_fcn(0, xi) + d1[3:] * basis_fcn(1, xi)
        d02 = d2[:3] * basis_fcn(0, xi) + d2[3:] * basis_fcn(1, xi)
        d01_xi = 0.5 * (d1[3:] - d1[:3])
        d02_xi = 0.5 * (d2[3:] - d2[:3])

        X0 = xpts[:3] * basis_fcn(0, xi) + xpts[3:] * basis_fcn(1, xi)
        X0_xi = 0.5 * (xpts[:3] - xpts[3:])
        n1 = fn1[:3] * basis_fcn(0, xi) + fn1[3:] * basis_fcn(1, xi)
        n2 = fn2[:3] * basis_fcn(0, xi) + fn2[3:] * basis_fcn(1, xi)
        n1_xi = 0.5 * (fn1[3:] - fn1[:3])
        n2_xi = 0.5 * (fn2[3:] - fn2[:3])

        # normalize and get transform T matrix
        t1 = X0_xi / np.linalg.norm(X0_xi)
        t2 = axis - np.dot(t1, axis) * t1
        t2 /= np.linalg.norm(t2)
        t3 = np.cross(t1, t2)
        T = np.array([list(t1), list(t2), list(t3)])

        # compute additional matrices
        Xd = np.array([list(X0_xi), list(n1), list(n2)])
        Xdinv = np.linalg.inv(Xd)
        detXd = np.linalg.det(Xd)
        XdinvT = Xdinv @ T
        u0d = np.array([list(u0_xi), list(d01), list(d02)])
        u0d_2 = u0d @ XdinvT
        u0x = T.T @ u0d_2
        e1 = np.array([1, 0, 0])
        s0 = np.dot(np.dot(XdinvT, e1), e1)
        sz1 = np.dot(np.dot(Xdinv, e1), n1_xi)
        sz2 = np.dot(np.dot(Xdinv, e1), n2_xi)
        d1x = s0 * np.dot(T.T, d01_xi - sz1 * u0_xi)
        d2x = s0 * np.dot(T.T, d2_xi - sz2 * u0_xi)

        # interp the tying strain is the same tying strain ety => gty for beam
        # transform tying strain to local coords
        e0ty = 2 * XdinvT[0,0] * ety

        # evaluate the strain
        strain = np.array([u0x[0], 0.5 * (d1x[2] - d2x[1]), d1x[0], d2x[0], e0ty[0], e0ty[1]])

        # evalate the stress
        
        stress = C @ strain

        # evaluate the strain sens




def add_mass_residual():
    pass

def get_stiffness_matrix():
    # use finite difference on add_residual to get Kelem
    # use add_st
    pass

def get_mass_matrix():
    # also use finite difference on the M * uddot residual term to get Melem
    # use add_mass_residual
    pass