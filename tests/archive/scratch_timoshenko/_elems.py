"""
Start an EB element from scatch based on the solution in https://link.springer.com/content/pdf/10.1007/978-1-4020-9200-8_10.pdf
which has an element and mass matrix.
"""

import numpy as np

def get_quad2(iquad):
    # https://en.wikipedia.org/wiki/Gaussian_quadrature
    if iquad == 0:
        xi = -1.0/np.sqrt(3)
    else:
        xi = 1.0/np.sqrt(3)
    return 0.5 * (1-xi), 0.5 * (1+xi)

def get_quad2_grad(iquad):
    if iquad == 0:
        xi = -1.0/np.sqrt(3)
    else:
        xi = 1.0/np.sqrt(3)
    return -0.5, 0.5

def TS_element_stiffness(a, E, I, ks, G, A):
    Ke = np.zeros((4,4))

    # term 1 - EIz * (dth/dx)^2
    # 2 point Gauss quadrature
    for iquad in range(2):
        N1d, N2d = get_quad2_grad(iquad)
        Nthd = np.array([0, N1d, 0, N2d]).reshape((4,1)) / a
        weight = 1.0
        temp = Nthd @ Nthd.T
        # print(F"{temp=}")
        Ke += E * I * a * weight * Nthd @ Nthd.T

    # term 2 - ks*G*A*(dw/dx-th)^2
    # 1 point Gauss quadrature to prevent shear locking
    weight = 2.0 # for 1 point Gauss quad
    xi = 0.0
    N1, N2 = 0.5 * (1-xi), 0.5 * (1+xi)
    N1d, N2d = -0.5, 0.5
    Nwd = 1/a * np.array([N1d, 0, N2d, 0]).reshape((4,1))
    Nth = np.array([0, N1, 0, N2])
    vec = Nwd - Nth
    temp2 = vec @ vec.T
    # print(F"{temp2=}")
    Ke += ks * G * A * a * weight * vec @ vec.T
    return Ke

def TS_distr_load(q, a):
    Fe = np.zeros((4,1))
    for iquad in range(2):
        N1, N2 = get_quad2(iquad)
        Nw = np.array([N1, 0, N2, 0]).reshape((4,1))
        Fe += q * Nw * a
    return Fe

def TS_element_mass(a, rho, A, I):
    Me = np.zeros((4,4))
    for iquad in range(2):
        N1, N2 = get_quad2(iquad)
        Nw = np.array([N1, 0, N2, 0]).reshape((4,1))
        Nth = np.array([0, N1, 0, N2]).reshape((4,1))
        weight = 1.0
        Me += rho * A * a * weight * Nw @ Nw.T
        Me += rho * I * a * weight * Nth @ Nth.T
    return Me