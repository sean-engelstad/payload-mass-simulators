import numpy as np
import sys, numpy as np, os, time
from payload_mass_sim import *

# t1 = t2 = 5e-3 # m
# t1 = t2 = 1e-2
L = 1.0 # m
material = Material.aluminum()

# technically J,Ip may be different and closed-form is sqrt(G*J/rho*Ip/L**2) but J = Ip here so don't need it

omega_fact = np.sqrt(material.G / L**2 / material.rho)
ind = np.array([i for i in range(1, 4+1)])
omega_axial = (2.0 * ind - 1) * np.pi / 2.0  * omega_fact
print(f"{omega_axial=}")

# ratio = omega_pred / omega_axial
# print(f"{ratio=}")