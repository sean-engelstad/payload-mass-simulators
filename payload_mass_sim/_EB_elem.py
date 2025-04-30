# we choose the hermite cubic polynomials as the basis functions of the element
import matplotlib.pyplot as plt
import numpy as np

# deprecated EB elemeent
# can't use this with tuning fork structure
# missing important rotational stiffness terms

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

def get_kelem_transverse(xscale=1, E=1, I=1):
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

def get_melem_transverse(xscale=1, rho=1, A=1):
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
    # return rho * A * L / 6 * np.array([[2,1],[1,2]])
    return rho * A * L / 2 * np.array([[1.0, 0.0],[0.0, 1.0]])

def get_kelem_torsion(G=1, J=1, L=1):
    """get torsion element stiffness matrix (without GJ/L scaling, here EA=1)"""
    return G * J / L * np.array([[1,-1], [-1,1]])

def get_melem_torsion(rho=1, Ip=1, L=1):
    """get axial element mass matrix (without rho*A*L/6 scaling)"""
    # consistent formulation which is more accurate for dynamic analysis / modal analysis
    # consistent mass matrix
    # return rho * A * J * L / 3.0 * np.array([[2,1],[1,2]])
    # lumped mass matrix
    return rho * Ip * L / 2 * np.array([[1.0, 0.0],[0.0, 1.0]])

def get_felem_axial(q0=1, L=1):
    """get element load vector"""
    return q0 * L * np.array([0.5]*2)

def get_felem_transverse(xscale=1.0):
    """get element load vector"""
    nquad = 4
    nbasis = 4
    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule4(iquad)
        for ibasis in range(nbasis):
            felem[ibasis] += weight * xscale * get_basis_fcn(ibasis, xi, xscale)
    return felem




# applying timoshenko beam theory to beam elements
# Defining the two basis functions for our elemental coordinates:
def get_basis_func_timoshenko(xi):
    ''' define the basis functions for the Timoshenko beam element '''
    psi1 = (1 + xi) / 2
    psi2 = (1 - xi) / 2
    dpsi1 = 0.5
    dpsi2 = -0.5
    return psi1, psi2, dpsi1, dpsi2

# Computing the element stiffness matrix:
def get_kelem_transverse_timoshenko(xscale=1, E=1, I=1, ks=5/6, G=1, A=1):
    ''' get elemental stiffness matrix for the Timoshenko beam element '''
    nquad = 4
    nbasis = 4
    Kelem_tim = np.zeros((nbasis, nbasis))
    J = xscale / 2
    D = np.array([
        [E * I, 0], 
        [0, ks * G * A]
    ])
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule4(iquad)
        psi1, psi2, dpsi1, dpsi2 = get_basis_func_timoshenko(xi)
        B = np.array([
            [0, dpsi1/J, 0, dpsi2/J],
            [dpsi1/J, -psi1, dpsi2/J, -psi2]
        ])
        # computhing stiffness matrix from D and B arrays
        Kelem_tim += weight * ((B.T @ D @ B) * J)
    
    # import matplotlib.pyplot as plt
    # plt.imshow(Kelem_tim)
    # plt.show()
    return Kelem_tim

# Computing the element mass matrix:
def get_melem_transverse_timoshenko(xscale=1, rho=1, A=1, I=1):
    ''' get elemental mass matrix for the Timoshenko beam element '''
    nquad = 4
    nbasis = 4
    Melem_tim = np.zeros((nbasis, nbasis))
    J = xscale / 2
    for iquad in range(nquad):
        xi , weight = get_quadrature_rule4(iquad)
        psi1, psi2, dpsi1, dpsi2 = get_basis_func_timoshenko(xi)
        H1 = np.array([psi1, 0, psi2, 0]).reshape((4,1))
        H2 = np.array([0, psi1, 0, psi2]).reshape((4,1))
        # computing the mass matrix from H1 and H2 arrays
        Melem_tim += weight * (rho * A * (H1 @ H1.T) + rho * I * (H2 @ H2.T)) * J

    # import matplotlib.pyplot as plt
    # plt.imshow(Melem_tim)
    # plt.show()
    return Melem_tim





