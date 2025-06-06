"""
based on TACS timoshenko beam element
    the TACSBeamElement.h file
    @author : Sean Engelstad
"""
import numpy as np
from _TS_utilities import *

N_VARS = 6
N_QUAD = 4
N_BASIS = 2

# using this excellent paper to get second derivatives with complex-step method (for speed)
# https://ancs.eng.buffalo.edu/pdf/ancs_papers/2008/complex_step08.pdf
# it extends to second order derivatives!

def get_CK(E, A, G, J, Iy, Iz, ky=5.0/6.0, kz=5.0/6.0):
    # the C input for get_strain_energy and get_stiffness_matrix
    return np.array([
        E * A,
        G * J,
        E * Iz,
        E * Iy,
        ky * G * A,
        kz * G * A,
    ])

def get_CM(rho, A, Iy, Iz):
    # the CM input for get_kinetic_energy and get_mass_matrix
    return np.array([
        rho * A,
        rho * Iz,
        rho * Iy,
    ])

def get_strain_energy(
    xpts:np.ndarray, # 2 * 3 = 6 entries
    qvars:np.ndarray, # 2 * 6 = 12 entries
    ref_axis:np.ndarray, # 3 entries
    CK:np.ndarray, # length 6 array of constitutive values
) -> np.ndarray: 

    fn1, fn2 = get_beam_node_normals(N_BASIS, xpts, ref_axis)

    # print(f"{fn1=}\n{fn2=}")

    d1 = compute_director(qvars, fn1)
    d2 = compute_director(qvars, fn2)

    ety = compute_tying_strains(xpts, fn1, fn2, qvars, d1, d2)

    Uenergy = 0.0

    for iquad in range(2):
        xi = -1.0 + 2.0 * iquad

        u0_xi = 0.5 * (qvars[6:] - qvars[:6])
        u0_xi = u0_xi[:3]
        d01 = d1[:3] * basis_fcn(0, xi) + d1[3:] * basis_fcn(1, xi)
        d02 = d2[:3] * basis_fcn(0, xi) + d2[3:] * basis_fcn(1, xi)
        d01_xi = 0.5 * (d1[3:] - d1[:3])
        d02_xi = 0.5 * (d2[3:] - d2[:3])

        X0 = xpts[:3] * basis_fcn(0, xi) + xpts[3:] * basis_fcn(1, xi)
        X0_xi = 0.5 * (xpts[3:] - xpts[:3])
        n1 = fn1[:3] * basis_fcn(0, xi) + fn1[3:] * basis_fcn(1, xi)
        n2 = fn2[:3] * basis_fcn(0, xi) + fn2[3:] * basis_fcn(1, xi)
        n1_xi = 0.5 * (fn1[3:] - fn1[:3])
        n2_xi = 0.5 * (fn2[3:] - fn2[:3])

        # normalize and get transform T matrix
        t1 = X0_xi / np.linalg.norm(X0_xi)
        t2 = ref_axis - np.dot(t1, ref_axis) * t1
        t2 /= np.linalg.norm(t2)
        t3 = np.cross(t1, t2)
        T = np.array([list(t1), list(t2), list(t3)])

        # compute additional matrices
        Xd = np.array([list(X0_xi), list(n1), list(n2)])
        Xdinv = np.linalg.inv(Xd)
        detXd = np.linalg.det(Xd)
        XdinvT = Xdinv @ T
        # print(f"{u0_xi=} {d01=} {d02=}")
        u0d = np.array([list(u0_xi), list(d01), list(d02)])
        u0d_2 = u0d @ XdinvT
        u0x = T.T @ u0d_2
        e1 = np.array([1, 0, 0])
        s0 = np.dot(np.dot(XdinvT, e1), e1)
        sz1 = np.dot(np.dot(Xdinv, e1), n1_xi)
        sz2 = np.dot(np.dot(Xdinv, e1), n2_xi)
        # print(f"{s0=} {d01_xi=} {sz1=} {u0_xi=}")
        d1x = s0 * np.dot(T.T, d01_xi - sz1 * u0_xi)
        d2x = s0 * np.dot(T.T, d02_xi - sz2 * u0_xi)

        # interp the tying strain is the same tying strain ety => gty for beam
        # transform tying strain to local coords
        e0ty = 2 * XdinvT[0,0] * ety

        # evaluate the strain
        # print(f"{u0x=} {d1x=} {d2x=} {e0ty=}")
        strain = np.array([u0x[0,0], 0.5 * (d1x[2] - d2x[1]), d1x[0], d2x[0], e0ty[0], e0ty[1]])

        # evalate the stress (assume no cross-couplings since centered at centroid here)
        stress = np.array([
            CK[0] * strain[0], # CK[0] = E * A
            CK[1] * strain[1], # CK[1] = G * J
            CK[2] * strain[2], # CK[2] = E * Iz
            CK[3] * strain[3], # CK[3] = E * Iy
            CK[4] * strain[4], # CK[4] = ky * G * A
            CK[5] * strain[5], # CK[5] = kz * G * A
        ])

        Uenergy += 0.5 * detXd * np.dot(stress, strain)
    return Uenergy
        
def get_stiffness_matrix(
    xpts:np.ndarray, # 2 * 3 = 6 entries
    qvars:np.ndarray, # 2 * 6 = 12 entries
    ref_axis:np.ndarray, # 3 entries
    CK:np.ndarray, # length 6 array of constitutive values
):
    # again using this paper to get the second derivatives, https://ancs.eng.buffalo.edu/pdf/ancs_papers/2008/complex_step08.pdf
    # complex-step formula for second derivatives is f''(x) = 2/h^2 * [f(x) - real(f(x+i*h))]
    Kelem = np.zeros((12,12))
    qvars0 = qvars.copy()
    h = 1e-5
    Ufunc = lambda q : get_strain_energy(xpts, q, ref_axis, CK)
    for ivar in range(12):
        # compute diagonals first
        ipert = np.zeros((12,))
        ipert[ivar] = h
        out = 2.0 / h**2 * (Ufunc(qvars) - np.real(Ufunc(qvars+ipert*1j)))
        Kelem[ivar,ivar] = 2.0 / h**2 * (np.real(Ufunc(qvars)) - np.real(Ufunc(qvars+ipert*1j)))

    # now off diagonals
    for ivar in range(12):
        for jvar in range(12):
            if ivar == jvar: continue # skip already did diagonals
            ijpert = np.zeros((12,))
            ijpert[ivar] = h
            ijpert[jvar] = h

            Kelem[ivar,jvar] = 1.0 / h**2 * (np.real(Ufunc(qvars)) - np.real(Ufunc(qvars + ijpert * 1j))) - 0.5 * (Kelem[ivar,ivar] + Kelem[jvar,jvar])
            Kelem[jvar,ivar] = Kelem[ivar,jvar]
    return Kelem

def get_kinetic_energy(
    xpts:np.ndarray, # 2 * 3 = 6 entries
    qvars:np.ndarray, # 2 * 6 = 12 entries
    ref_axis:np.ndarray, # 3 entries
    CM:np.array, # 3 entry mass data vector
) -> np.ndarray: 

    if (np.linalg.norm(np.imag(qvars)) != 0.0) and not(np.linalg.norm(np.imag(qvars)) > 1.0e-5):
        ipert = np.argmax(np.imag(qvars))
        # print(f"{qvars=} ----------")
    
    fn1, fn2 = get_beam_node_normals(N_BASIS, xpts, ref_axis)

    # let's just use the qvars as if they are director rates and as if they are speeds
    # since quadratic, Melem is indep. of the actual speeds, just need to compute it
    d1dot = compute_director(qvars, fn1)
    d2dot = compute_director(qvars, fn2)

    Tenergy = 0.0

    for iquad in range(2):
        xi = -1.0 + 2.0 * iquad

        u0_dot = qvars[:6] * basis_fcn(0, xi) + qvars[6:] * basis_fcn(1, xi)
        u0_dot = u0_dot[:3]
        d01_dot = d1dot[:3] * basis_fcn(0, xi) + d1dot[3:] * basis_fcn(1, xi)
        d02_dot = d2dot[:3] * basis_fcn(0, xi) + d2dot[3:] * basis_fcn(1, xi)

        # print(f"{u0_dot=} {d01_dot=}")

        # get detXd from coords ------------------------
        X0 = xpts[:3] * basis_fcn(0, xi) + xpts[3:] * basis_fcn(1, xi)
        X0_xi = 0.5 * (xpts[3:] - xpts[:3])
        n1 = fn1[:3] * basis_fcn(0, xi) + fn1[3:] * basis_fcn(1, xi)
        n2 = fn2[:3] * basis_fcn(0, xi) + fn2[3:] * basis_fcn(1, xi)
        Xd = np.array([list(X0_xi), list(n1), list(n2)])
        Xdinv = np.linalg.inv(Xd)
        detXd = np.linalg.det(Xd)
        # end of compute detXd --------------------------

        loc_Te = 0.0
        loc_Te += CM[0] * np.dot(u0_dot, u0_dot) # CM[0] = rho*A
        loc_Te += CM[1] * np.dot(d01_dot, d01_dot) # CM[1] = rho * Iz
        loc_Te += CM[2] * np.dot(d02_dot, d02_dot) # CM[2] = rho * Iy

        dot1 = np.dot(u0_dot, u0_dot)
        dot2 = np.dot(d01_dot, d01_dot)
        dot3 = np.dot(d02_dot, d02_dot)
        # if (np.linalg.norm(np.imag(qvars)) != 0.0) and not(np.linalg.norm(np.imag(qvars)) > 1.0e-5) and iquad == 1:
        #     print(F"{dot1=}, {dot2=}, {dot3=}")
        # print(F"{dot1=}, {dot2=}, {dot3=}")

        # print(F"{loc_Te=}\n{detXd=}\n{CM[0]}")

        # then add into total energy
        Tenergy += loc_Te * 0.5 * detXd
    # if (np.linalg.norm(np.imag(qvars)) != 0.0) and not(np.linalg.norm(np.imag(qvars)) > 1.0e-5):
    #     print(f"{Tenergy=}")
    return Tenergy
    

def get_mass_matrix(
    xpts:np.ndarray, # 2 * 3 = 6 entries
    qvars:np.ndarray, # 2 * 6 = 12 entries
    ref_axis:np.ndarray, # 3 entries
    CM:np.array, # 3 entry mass data vector
):
    # again using this paper to get the second derivatives, https://ancs.eng.buffalo.edu/pdf/ancs_papers/2008/complex_step08.pdf
    # complex-step formula for second derivatives is f''(x) = 2/h^2 * [f(x) - real(f(x+i*h))]
    Melem = np.zeros((12,12))
    qvars0 = qvars.copy()
    h = 1e-6
    Tfunc = lambda q : get_kinetic_energy(xpts, q, ref_axis, CM)
    for ivar in range(12):
        # compute diagonals first
        ipert = np.zeros((12,))
        ipert[ivar] = h
        Melem[ivar,ivar] = 2.0 / h**2 * (np.real(Tfunc(qvars)) - np.real(Tfunc(qvars+ipert*1j)))
        out = 2.0 / h**2 * (np.real(Tfunc(qvars)) - np.real(Tfunc(qvars+ipert*1j)))
        # print(F"{ivar=} {out=}")

    # now off diagonals
    for ivar in range(12):
        for jvar in range(12):
            if ivar == jvar: continue # skip already did diagonals
            ijpert = np.zeros((12,))
            ijpert[ivar] = h
            ijpert[jvar] = h

            Melem[ivar,jvar] = 1.0 / h**2 * (np.real(Tfunc(qvars)) - np.real(Tfunc(qvars + ijpert * 1j))) - 0.5 * (Melem[ivar,ivar] + Melem[jvar,jvar])
            Melem[jvar,ivar] = Melem[ivar,jvar]
    return Melem
