# test the strain energies and kinetic energies of the elements
import numpy as np
import matplotlib.pyplot as plt
from _elems import TS_element_stiffness, TS_element_mass

# choose EI/Le^2=1
E = 1.0
I = 1.0
Le = 1.0

# choose ks*G*A = 0.1 or something smaller
KSGA = 0.1
ks = 1.0
G = 1.0
A = KSGA

# get kelem
Kelem = TS_element_stiffness(E, I, ks, G, A, Le)

# assuming th = 1 + xi, w=0 - strain energy approx
uelem = np.array([0, 0, 0, 2]).reshape((4,1))
Uelem = uelem.T @ Kelem @ uelem * 0.5
print(f"{Uelem=}")

# now check against analytic strain energy here
Uelem_truth = 0.5 * 2 * E * I / Le**2 * Le + 0.5 * 8.0/3.0 * Le * KSGA
print(f"{Uelem_truth=}")

# ok great, now check the mass
