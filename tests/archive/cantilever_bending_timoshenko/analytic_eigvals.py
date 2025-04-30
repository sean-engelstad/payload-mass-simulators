import numpy as np
import sys, numpy as np, os, time
from payload_mass_sim import *

# https://mooseframework.inl.gov/modules/solid_mechanics/1d_elastic_waves.html#:~:text=The%20analytic%20eigenvalues%2C%20omega_n%20%CF%89n%2C%20are%20given%20by,length%2C%20and%20k_n%20kn%20are%20the%20wave%20numbers.

# t1 = t2 = 5e-3 # m
t1 = t2 = 1e-1
L = 1.0 # m
material = Material.aluminum()
I = t1 * t2**3 / 12.0
A = t1 * t2
timoshenko = True

# Euler-bernoulli beam
omega_fact = np.sqrt(material.E * I / material.rho / A / L**4)
k_vec = np.array([1.875, 4.694, 7.855])
omega_bending = k_vec**2 * omega_fact
print(f"{omega_bending=}")

axial = np.sqrt(material.E / material.rho) / L
print(f'{axial=}')

torsion = np.sqrt(material.G / material.rho) / L
print(f'{torsion=}')

# Timoshenko beam
k_vec = np.array([1.875, 4.694, 7.855]) # mode shape constants
mode_nums = np.arange(1, len(k_vec)+1)   # n = 1, 2, 3,...
print(f"{mode_nums=}")
correction_denominator = 1 + (mode_nums**2) * np.pi**2 * t1**2 * material.E / (material.k_s * material.G * L**2)
omega_bending_timoshenko = omega_bending / np.sqrt(correction_denominator)
print(f"{omega_bending_timoshenko=}")

# bending_timoshenko_freq = np.sqrt(omega_bending_timoshenko)
# print(f"{bending_timoshenko_freq=}")