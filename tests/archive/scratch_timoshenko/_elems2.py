import numpy as np

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

def get_quadrature_rule2(iquad):
    # 3rd order
    rt3 = 1.0/np.sqrt(3)
    if iquad == 0:  
        return -rt3, 1.0
    else:
        return rt3, 1.0
    
def get_quadrature_rule4(iquad):
    # 4th order
    gauss_points = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])
    gauss_weights = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
    return gauss_points[iquad],  gauss_weights[iquad]

def get_basis_func_timoshenko(xi):
    ''' define the basis functions for the Timoshenko beam element '''
    psi1 = (1 + xi) / 2
    psi2 = (1 - xi) / 2
    dpsi1 = 0.5
    dpsi2 = -0.5
    return psi1, psi2, dpsi1, dpsi2

# Computing the element stiffness matrix:
def TS_element_stiffness(xscale=1, E=1, I=1, ks=5/6, G=1, A=1):
    ''' get elemental stiffness matrix for the Timoshenko beam element '''
    nquad = 4
    nbasis = 4
    Kelem_tim = np.zeros((nbasis, nbasis))
    J = xscale / 2
    D = np.array([
        [E * I, 0], 
        [0, ks * G * A]
    ])

    mode = 2
    if mode == 1: # equally integrated
        for iquad in range(nquad):
            xi, weight = get_quadrature_rule4(iquad)
            psi1, psi2, dpsi1, dpsi2 = get_basis_func_timoshenko(xi)
            B = np.array([
                [0, dpsi1/J, 0, dpsi2/J],
                [dpsi1/J, -psi1, dpsi2/J, -psi2]
            ])
            # computhing stiffness matrix from D and B arrays
            Kelem_tim += weight * ((B.T @ D @ B) * J)
    elif mode == 2: # reduced integrated
        for iquad in range(4):
            xi, weight = get_quadrature_rule4(iquad)
            psi1, psi2, dpsi1, dpsi2 = get_basis_func_timoshenko(xi)
            B1 = np.array([0, dpsi1/J, 0, dpsi2/J]).reshape((4,1))
            # computhing stiffness matrix from D and B arrays
            Kelem_tim += weight * (E * I * B1 @ B1.T) * J
        for iquad in range(3):
            xi, weight = get_quadrature_rule3(iquad)
            psi1, psi2, dpsi1, dpsi2 = get_basis_func_timoshenko(xi)
            B2 = np.array([dpsi1/J, -psi1, dpsi2/J, -psi2]).reshape((4,1))
            # computhing stiffness matrix from D and B arrays
            Kelem_tim += weight * (ks * G * A * B2 @ B2.T) * J

    # import matplotlib.pyplot as plt
    # plt.imshow(Kelem_tim)
    # plt.show()
    return Kelem_tim

# Computing the element mass matrix:
def TS_element_mass(xscale=1, rho=1, A=1, I=1):
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




