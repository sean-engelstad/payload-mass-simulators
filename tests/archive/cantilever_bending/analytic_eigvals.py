import numpy as np
import sys, numpy as np, os, time
from payload_mass_sim import *

# https://mooseframework.inl.gov/modules/solid_mechanics/1d_elastic_waves.html#:~:text=The%20analytic%20eigenvalues%2C%20omega_n%20%CF%89n%2C%20are%20given%20by,length%2C%20and%20k_n%20kn%20are%20the%20wave%20numbers.

# t1 = t2 = 5e-3 # m
t1 = t2 = 1e-2
L = 1.0 # m
material = Material.aluminum()
I = t1 * t2**3 / 12.0
A = t1 * t2

omega_fact = np.sqrt(material.E * I / material.rho / A / L**4)
k_vec = np.array([1.875, 4.694, 7.855])
omega_bending = k_vec**2 * omega_fact
print(f"{omega_bending=}")

axial = np.sqrt(material.E / material.rho) / L
print(f'{axial=}')

torsion = np.sqrt(material.G / material.rho) / L
print(f'{torsion=}')