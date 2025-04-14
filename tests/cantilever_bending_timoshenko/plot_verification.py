import numpy as np
import matplotlib.pyplot as plt
import niceplots
from payload_mass_sim import *

L = 1.0 # m
material = Material.aluminum()

def get_analytic_timoshenko_eigval(thick, timoshenko=True):
    t1 = t2 = thick
    I = t1 * t2**3 / 12.0
    A = t1 * t2
    omega_EB = np.array([1.875, 4.694, 7.855])**2 * np.sqrt(material.E * I / material.rho / A / L**4)
    TS_denom = 1 + (np.array([1,2,3])**2) * np.pi**2 * t1**2 * material.E / (material.k_s * material.G * L**2)
    omega_TS = omega_EB / np.sqrt(TS_denom)
    if timoshenko:
        return omega_TS[0]
    else:
        return omega_EB[0]

def get_local_timoshenko_eigval(thick, timoshenko=True):
    tree = TreeData(
        tree_start_nodes=[0],
        tree_directions=[0],
        nelem_per_comp=50
    )
    t1 = t2 = thick
    ncomp = 1
    init_design = np.array([L, t1, t2]*ncomp)
    num_dvs = init_design.shape[0]
    beam3d = Beam3DTree(material, tree, timoshenko=timoshenko)
    beam3d._bend1_mult = 1.0
    beam3d._bend2_mult = 1e4
    freqs = beam3d.get_frequencies(init_design)
    return freqs[0]

thicks = np.geomspace(1e-2, 5e-1, 10)
analytic_EB = np.array([get_analytic_timoshenko_eigval(thick, False) for thick in thicks])
analytic_TS = np.array([get_analytic_timoshenko_eigval(thick, True) for thick in thicks])
local_EB = np.array([get_local_timoshenko_eigval(thick, False) for thick in thicks])
local_TS = np.array([get_local_timoshenko_eigval(thick, True) for thick in thicks])

plt.style.use(niceplots.get_style())
plt.margins(x=0.05, y=0.05)
plt.plot(thicks, analytic_TS, 'o-', label="analytic-TS")
plt.plot(thicks, analytic_EB, 'o-', label="analytic-EB")
plt.plot(thicks, local_EB, 'o--', label="local-EB")
plt.plot(thicks, local_TS, 'o--', label="local-TS")
plt.legend()
# plt.show()
plt.xscale('log')
plt.savefig("eb_ts_compare.png", dpi=400)