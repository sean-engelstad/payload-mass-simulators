import sys, numpy as np, os, time
sys.path.append("beam-fea")
from solver_3D import *
from pyoptsparse import SNOPT, Optimization

if __name__ == "__main__":

    material = Material.aluminum()

    # test out the tree data structure
    # TODO : modify tree start nodes and directions graph
    #    in order to have level 2 optimization
    tree = TreeData(
        tree_start_nodes=[0] + [1] * 4 + [2,3,4,5],
        tree_directions=[5] + [0,1,2,3] + [5]*4,
        nelem_per_comp=10
    )
    # init_lengths = [1.0]*tree.ncomp
    # xpts = tree.get_xpts(init_lengths)
    # tree.plot_mesh(xpts)

    # initial design variables
    ncomp = tree.ncomp
    init_design = np.array([0.3, 5e-3, 5e-3]*ncomp)
    num_dvs = init_design.shape[0]

    opt_design = [0.08125551, 0.02211063, 0.07895405, 0.14713621, 0.01711272,
       0.05670854, 0.14552369, 0.00420714, 0.02302042, 0.17597928,
       0.04472327, 0.0450355 , 0.11794632, 0.05531567, 0.00601055,
       0.08715245, 0.00400008, 0.00345651, 0.14678793, 0.0039524 ,
       0.00351281, 0.12807156, 0.00403881, 0.00427038, 0.16456358,
       0.00351356, 0.00346978]
    opt_design = np.array(opt_design)

    beam3d = Beam3DTree(material, tree)

    # now build 3D beam solver and solve the eigenvalue problem
    # freqs = beam3d.get_frequencies(init_design)
    freqs = beam3d.get_frequencies(opt_design)
    print(f"{freqs=}")
    beam3d.plot_eigenmodes(
        nmodes=10,
        show=False,
        def_scale=1.0
    )

    # now also test linear static
    beam3d._solve_static(opt_design)
    beam3d.plot_static(
        show=False,
        def_scale=1e1,
    )
    

    