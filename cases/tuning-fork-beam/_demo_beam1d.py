# now let's test this out and visualize it
import sys, numpy as np
sys.path.append("beam-fea")
from solver_1D import Transverse1DBeam

E = 2e7; b = 4e-3; L = 1; rho = 1
qmag = 2e-2
ys = 4e5

# scaling inputs
# if rho_KS was too high like 500, then the constraint was too nonlinear or close to max and optimization failed
rho_KS = 50.0 # rho = 50 for 100 elements, used 500 later
nxe = num_elements = int(3e2) #100, 300, 1e3

hvec = np.array([1e-3] * nxe)

beam_fea = Transverse1DBeam(nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False)
beam_fea.solve_forward(hvec)
beam_fea.plot_disp()