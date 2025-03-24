import sys, numpy as np, os, time
sys.path.append("beam-fea")
from solver_3D import *

# init design
ncomp = 9
init_design = np.array([0.3, 5e-3, 5e-3]*ncomp)

# optimal design
opt_design = [0.08125551, 0.02211063, 0.07895405, 0.14713621, 0.01711272,
       0.05670854, 0.14552369, 0.00420714, 0.02302042, 0.17597928,
       0.04472327, 0.0450355 , 0.11794632, 0.05531567, 0.00601055,
       0.08715245, 0.00400008, 0.00345651, 0.14678793, 0.0039524 ,
       0.00351281, 0.12807156, 0.00403881, 0.00427038, 0.16456358,
       0.00351356, 0.00346978]
opt_design = np.array(opt_design)

material = Material.aluminum()
tree = TreeData(
    tree_start_nodes=[0] + [1]*4 + [2,3,4,5],
    tree_directions=[5] + [0,1,2,3] + [5]*4,
    nelem_per_comp=10
)
beam3d = Beam3DTree(material, tree)

# plot modes for initial design
freqs_init = beam3d.get_frequencies(init_design)
print(f"{freqs_init=}")
nmodes = 5
beam3d.plot_eigenmodes(
    nmodes=nmodes,
    show=True,
    def_scale=0.5,
    file_prefix="init"
)

# plot modes for optimal design
freqs_opt = beam3d.get_frequencies(opt_design)
print(f"{freqs_opt=}")
nmodes = 5
beam3d.plot_eigenmodes(
    nmodes=nmodes,
    show=True,
    def_scale=0.5,
    file_prefix="opt"
)