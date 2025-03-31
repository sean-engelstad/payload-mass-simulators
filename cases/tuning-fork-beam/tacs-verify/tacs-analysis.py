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

bdfFile = os.path.join(os.path.dirname(__file__), "tuning-fork.bdf")
# Load BDF file
FEAAssembler = pyTACS(bdfFile, comm, options=structOptions)

# Material properties
rho = 2700.0  # density kg/m^3
E = 70.0e9  # Young's modulus (Pa)
nu = 0.3  # Poisson's ratio
ys = 270.0e6  # yield stress

# Shell thickness
A = 0.1  # m
Iz = 0.2  # m
Iy = 0.3  # m
J = 0.4


# Callback function used to setup TACS element objects and DVs
def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.BasicBeamConstitutive(
        prop, A=A, Iy=Iy, Iz=Iz, J=J, ky=1000, kz=1000
    )
    # exit()

    print(F"{compID=} {compDescript=}", flush=True)

    # refAxis is perp to beam and is Iy direc
    if "Base" in compDescript or "Vert" in compDescript:
        # print("in here")
        refAxis = np.array([0.0, 1.0, 0.0])
    elif "Lateral" in compDescript:
        refAxis = np.array([0.0, 0.0, 1.0])

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

MP = FEAAssembler.createModalProblem("modal", sigma=10.0, numEigs=10)
MP.setOption("printLevel", 2)
MP.solve()

if not os.path.exists("_buckling"):
    os.mkdir("_buckling")
MP.writeSolution(outputDir="_buckling")