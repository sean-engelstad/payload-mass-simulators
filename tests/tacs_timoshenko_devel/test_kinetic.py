# check the kinetic energy inputs make sense

import numpy as np
import matplotlib.pyplot as plt
from _TS_elem import *

# aluminum beam
E, nu, rho = 70e9, 0.3, 1.225e3
G = E / 2.0 / (1+nu)

# geometry
hL = 1e-1
t1, t2, L = hL, hL, 1.0
Iy, Iz = t1**3 * t2 / 12.0, t2**3 * t1 / 12.0
ks = 5/6 # shear correction of rectangular cross-section
ky, kz = ks, ks
J, A = Iy + Iz, t1 * t2

CM = get_CM(rho, A, Iy, Iz)

# now compute some kinetic energies
# 1 - rho*A scaled energy
xpts = np.array([0.0] * 3 + [1.0, 0.0, 0.0])
qvars = np.array([1.0] + [0.0] * 5 + [1.0] + [0.0]*5)
ref_axis = np.array([0.0, 1.0, 0.0])
T = get_kinetic_energy(xpts, qvars, ref_axis, CM)
Tref = 0.5 * rho * A * L * 1.0
print(f"{T=}\n{Tref=}")  

# 2 - rho * Iz scaled energy
qvars2 = np.array([0.0] * 3 + [0.0, 1.0, 0.0] + [0.0]*3 + [0.0, 1.0, 0.0])
T2 = get_kinetic_energy(xpts, qvars2, ref_axis, CM)
print(f"{T2=}")