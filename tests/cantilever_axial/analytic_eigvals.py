import numpy as np
import sys, numpy as np, os, time
from payload_mass_sim import *

# t1 = t2 = 5e-3 # m
t1 = t2 = 1e-2
L = 1.0 # m
material = Material.aluminum()
I = t1 * t2**3 / 12.0
A = t1 * t2

omega_fact = np.sqrt(material.E / L**2 / material.rho)
ind = np.array([i for i in range(1, 4+1)])
omega_axial = (2.0 * ind - 1) * np.pi / 2.0  * omega_fact
print(f"{omega_axial=}")

# lumped mass matrix
omega_pred = np.array([ 11310.93316928,  33929.93297564,  56540.55574854,  79137.25182283,
       101714.45829449])[:4]
# consistent mass matrix
# omega_pred = np.array([ 19596.30775867,  58909.86407004,  98586.93605203, 138871.92378285,
#        163428.31030404])[:4]

ratio = omega_pred / omega_axial
print(f"{ratio=}")