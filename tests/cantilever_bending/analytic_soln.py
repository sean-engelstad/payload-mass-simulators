from payload_mass_sim import *
import numpy as np

material = Material.aluminum()

E = material.E
rho = material.rho

t1 = t2 = 1e-3 # m
L = 1.0 # m

# I1 = t1 * t2**3 / 12.0
I1 = 0
I2 = t2 * t1**3 / 12.0
I = I1 + I2


freq = np.array([3.5156, 6.268, 17.456]) * 1.0/L**2 * np.sqrt(E*I / rho)
print(f"{freq=}")

print(f"{rho=} {E=}")

