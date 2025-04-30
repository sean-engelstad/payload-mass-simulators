"""
6 noded beam model 1 meter long in x direction.
We apply a tip load in the z direction and clamp it at the root.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================
from pprint import pprint
from mpi4py import MPI
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import constitutive, elements, functions, pyTACS

comm = MPI.COMM_WORLD

# Instantiate FEAAssembler
structOptions = {}

bdfFile = os.path.join(os.path.dirname(__file__), "cantilever.bdf")
# Load BDF file
FEAAssembler = pyTACS(bdfFile, comm, options=structOptions)

# Material properties
rho = 2700.0  # density kg/m^3
E = 70.0e9  # Young's modulus (Pa)
nu = 0.3  # Poisson's ratio
ys = 270.0e6  # yield stress

sigma = 1e-2 # eigenvalue guess

# Shell thickness
# A = 0.1  # m
# Iz = 0.2  # m
# Iy = 0.3  # m
# J = 0.4
# b = h = 5e-3
b = h = 1e-2
A = b * h
Iy = Iz = b * h**3 / 12.0
J = 2 * Iz

# J *= 1e3

# kTransverse = 1e3 # orig 1000, but want to make it high so becomes Euler-Bernoulli
# kTransverse *= 1e2
kTransverse = 5.0 / 3.0

# Callback function used to setup TACS element objects and DVs
def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.BasicBeamConstitutive(
        prop, A=A, Iy=Iy, Iz=Iz, J=J, ky=kTransverse, kz=kTransverse
    )
    # exit()

    print(F"{compID=} {compDescript=}", flush=True)

    # refAxis is perp to beam and is Iy direc
    refAxis = np.array([1.0, 0.0, 0.0])

    print(F"{compDescript=} {refAxis=}", flush=True)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = elements.BeamRefAxisTransform(refAxis)
    # transform = None
    elem = elements.Beam2(transform, con)
    return elem

# elemCallBack = None

# Set up elements and TACS assembler
FEAAssembler.initialize(elemCallBack)

# ==============================================================================
# Setup static problem
# ==============================================================================
# Static problem
evalFuncs = ["mass", "ks_vmfailure"]

num_eig = 10
MP = FEAAssembler.createModalProblem("modal", sigma=10.0, numEigs=num_eig)
MP.setOption("printLevel", 2)
MP.solve()

if not os.path.exists("_modal"):
    os.mkdir("_modal")
MP.writeSolution(outputDir="_modal")

# get the square-root eigvals for natural frequencies
funcs = {}
evalFuncs = [f"eigsm.{i}" for i in range(num_eig)]
MP.evalFunctions(funcs, evalFuncs)
# print(f"{funcs=}")
eigvals = [funcs[key] for key in funcs]
nat_freq = np.sqrt(np.array(eigvals))
print(f"{nat_freq=}")
