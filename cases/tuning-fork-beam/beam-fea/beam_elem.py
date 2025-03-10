# we choose the hermite cubic polynomials as the basis functions of the element
import matplotlib.pyplot as plt
import numpy as np

def hermite_cubic_polynomials_1d(ibasis):
    # node 1 is xi = -1, node 2 is xi = 1
    if ibasis == 0: # w for node 1
        return [0.5, -0.75, 0.0, 0.25]
    elif ibasis == 1: # dw/dx for node 1
        return [0.25, -0.25, -0.25, 0.25]
    elif ibasis == 2: # w for node 2
        return [0.5, 0.75, 0.0, -0.25]
    elif ibasis == 3: # dw/dx for node 2
        return [-0.25, -0.25, 0.25, 0.25]
    
def eval_polynomial(poly_list, value):
    poly_list_arr = np.array(poly_list)
    var_list_arr = np.array([value**(ind) for ind in range(len(poly_list))])
    return np.dot(poly_list_arr, var_list_arr)

def hermite_cubic_1d(ibasis, xi):
    poly_list = hermite_cubic_polynomials_1d(ibasis)
    return eval_polynomial(poly_list, xi)

def plot_hermite_cubic():
    xi_vec = np.linspace(-1, 1, 100)
    for ibasis in range(4):
        poly = hermite_cubic_polynomials_1d(ibasis)
        h_vec = np.array([eval_polynomial(poly, xi) for xi in xi_vec])
        plt.plot(xi_vec, h_vec, label=f"phi_{ibasis}")
    plt.legend()
    plt.show()

# and the following quadrature rule for 1D elements
def get_quadrature_rule3(iquad):
    # 3rd order
    rt35 = np.sqrt(3.0/5.0)
    if iquad == 0:
        return -rt35, 5.0/9.0
    elif iquad == 1:
        return 0.0, 8.0/9.0
    elif iquad == 2:
        return rt35, 5.0/9.0
    
def get_quadrature_rule4(iquad):
    # 4th order
    gauss_points = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])
    gauss_weights = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
    return gauss_points[iquad],  gauss_weights[iquad]
    
# this is how to compute the element stiffness matrix:
def get_hess(ibasis, xi, xscale):
    xi_poly = hermite_cubic_polynomials_1d(ibasis)

    dphi_xi2_poly = [2.0 * xi_poly[-2], 6.0 * xi_poly[-1]]
    dphi_xi2 = eval_polynomial(dphi_xi2_poly, xi)
    dphi_dx2 = 1.0/xscale**2 * dphi_xi2
    return dphi_dx2

def get_basis_fcn(ibasis, xi, xscale):
    xi_poly = hermite_cubic_polynomials_1d(ibasis)
    return eval_polynomial(xi_poly, xi)

def get_kelem_transverse(xscale, E=1, I=1):
    """get element stiffness matrix (without EI scaling)"""
    nquad = 4
    nbasis = 4
    Kelem = np.zeros((nbasis, nbasis))
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule4(iquad)
        for i in range(nbasis):
            for j in range(nbasis):
                Kelem[i,j] += weight * xscale * get_hess(i, xi, xscale) * get_hess(j, xi, xscale)
    return E * I * Kelem

def get_melem_transverse(xscale, rho=1, A=1):
    """get element mass matrix (without rho*A scaling)"""
    nquad = 4 # needed 4th order quadrature to get accurate eigenvalues with beams
    nbasis = 4
    # Melem = rho*A * integral(phi_i * phi_j) dx
    Melem = np.zeros((nbasis, nbasis))
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule4(iquad)
        for i in range(nbasis):
            for j in range(nbasis):
                Melem[i,j] += weight * xscale * get_basis_fcn(i, xi, xscale) * get_basis_fcn(j, xi, xscale)
    return rho * A * Melem

def get_kelem_axial(E=1, A=1, L=1):
    """get axial element stiffness matrix"""
    return E * A / L * np.array([[1,-1], [-1,1]])

def get_melem_axial(rho=1, A=1, L=1):
    """get axial element mass matrix"""
    # consistent formulation which is more accurate for dynamic analysis / modal analysis
    return rho * A * L / 6 * np.array([[2,1],[1,2]])

def get_kelem_torsion(G=1, J=1, L=1):
    """get torsion element stiffness matrix (without GJ/L scaling, here EA=1)"""
    return G * J / L * np.array([[1,-1], [-1,1]])

def get_melem_torsion(rho=1, A=1, J=1, L=1):
    """get axial element mass matrix (without rho*A*L/6 scaling)"""
    # consistent formulation which is more accurate for dynamic analysis / modal analysis
    return rho * A * J * L / 3.0 * np.array([[2,1],[1,2]])

def get_felem(xscale):
    """get element load vector"""
    nquad = 3
    nbasis = 4
    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule4(iquad)
        for ibasis in range(nbasis):
            felem[ibasis] += weight * xscale * get_basis_fcn(ibasis, xi, xscale)
    return felem