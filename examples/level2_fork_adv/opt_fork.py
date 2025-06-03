import sys, numpy as np, os, time
from payload_mass_sim import *
from pyoptsparse import SNOPT, Optimization
import argparse
from _utils import test

def parse_args():
    parser = argparse.ArgumentParser(description="Constraint toggles")

    parser.add_argument('--freq', action='store_true', help='Use frequency constraints')
    parser.add_argument('--mass', action='store_true', help='Use mass constraints')
    parser.add_argument('--cg', action='store_true', help='Use center of gravity constraints')
    parser.add_argument('--failure', action='store_true', help='Use failure constraints')
    parser.add_argument('--level', type=int, default=1, help='Level of tuning fork (only goes up to 2 at the moment)')
    parser.add_argument('--output', type=str, default="level1", help="output directory")
    parser.add_argument('--reset', action='store_true', help="reset output directory")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif not(args.reset):
        raise RuntimeError(f"Don't overwrite to prev output directory, choose diff name")

    material = Material.aluminum()

    def loc_dv_list(length, thick=5e-3):
        # starting dv list
        return ([length] + [thick]*4 + [0.1, 0.5])

    # tree data structure
    # -------------------
    # directions [0,1,2,3,4,5] equiv to x-,x+,y-,y+,z-,z+

    # base and level 1
    init_design = loc_dv_list(0.3) * 9
    tree_start_nodes = [0] + [1]*4 + [2,3,4,5]
    tree_directions = [5] + [0,1,2,3] + [5]*4

    Llittle = 0.2 # 0.1

    # include_level2 = True
    include_level2 = False # just level1

    if include_level2:

        # level 2, x- branch
        init_design += loc_dv_list(Llittle) * 8
        tree_start_nodes += [6]*4 + [10,11,12,13]
        tree_directions += [0,1,2,3] + [5]*4

        # level 2, x+ branch
        init_design += loc_dv_list(Llittle) * 8
        tree_start_nodes += [7]*4 + [18,19,20,21]
        tree_directions += [0,1,2,3] + [5]*4

        # level 2, y- branch
        init_design += loc_dv_list(Llittle) * 8
        tree_start_nodes += [8]*4 + [26,27,28,29]
        tree_directions += [0,1,2,3] + [5]*4

        # level 2, y+ branch
        init_design += loc_dv_list(Llittle) * 8
        tree_start_nodes += [9]*4 + [34,35,36,37]
        tree_directions += [0,1,2,3] + [5]*4

    tree = TreeData(
        tree_start_nodes=tree_start_nodes,
        tree_directions=tree_directions,
        nelem_per_comp=5 
    )
    init_design = np.array(init_design)

    # tree.set_assembler_data(7, material.E)
    # tree.centroid_FD_test(init_design, h=1e-4)
    # exit()

    # for derivatives testing, want to move somewhat away from equal thickness design prob
    # init_design += np.random.rand(init_design.shape[0]) * 1e-2

    ncomp = tree.ncomp
    num_dvs = init_design.shape[0]
    inertial_data = InertialData([1, 0, 0])

    beam3d = BeamAssemblerAdvanced(material, tree, inertial_data, rho_KS=3.0)

    # test(beam3d, init_design, demo=True, deriv=False)

    # optimization
    # -------------------------------------

    nmodes = 4 # we're targeting the first 4 eigenvalues for now
    target_eigvals = np.array([24.3, 29.4, 99.5, 173.6])
    target_mass = 176.407 # kg
    target_centroid = np.array([0.0104782366, 0.005086985, 0.417962588]) #m

    #nmodes = 6 # we're targeting the first 4 eigenvalues for now
    #target_eigvals = np.array([24.3, 29.4, 99.5, 173.6, 1, 1])

    xarr = init_design

    def get_functions(x_dict):
        xlist = x_dict["vars"]
        xarr = np.array([float(_) for _ in xlist])

        # print(f"{xarr=}")
        funcs = {}

        if args.freq:
            freqs = beam3d.get_frequencies(xarr)
            for imode in range(nmodes):
                funcs[f'freq{imode}'] = freqs[imode]

        if args.mass:
            mass = beam3d.get_mass(xarr)
            funcs['mass'] = mass
        
        if args.cg:
            centroid = tree.get_centroid(xarr)
            for i in range(3):
                funcs[f'centroid{i}'] = centroid[i]

        if args.failure:
            beam3d.solve_static(xarr)
            fail_index = beam3d.get_failure_index(xarr)
            funcs['failure'] = fail_index

        funcs['dummyobj'] = 0.0

        # writeout a current opt-status.txt file
        hdl = open(f"{args.output}/opt-status.txt", mode='w')
        hdl.write("funcs:\n")
        funcs_keys = list(funcs.keys())
        for i,key in enumerate(funcs_keys):
            if "freq" in key:
                hdl.write(f"\tfunc {key} = {funcs[key]:.4e}, target {target_eigvals[i]}\n")
            elif 'mass' in key:
                hdl.write(f"\tfunc {key} = {mass:.4e}, target {target_mass}\n")
            elif 'centroid0' in key:
                hdl.write(f"\tfunc {key} = {centroid[0]:.4e}, target {target_centroid[0]}\n")
            elif 'centroid1' in key:
                hdl.write(f"\tfunc {key} = {centroid[1]:.4e}, target {target_centroid[1]}\n")
            elif 'centroid2' in key:
                hdl.write(f"\tfunc {key} = {centroid[2]:.4e}, target {target_centroid[2]}\n")
            elif 'failure' in key:
                hdl.write(f"\tfunc {key} = {fail_index:.4e} <= 1.0\n")
            else:
                hdl.write(f"\tfunc {key} = {funcs[key]:.4e}\n")
        hdl.write("vars:\n")

        dv_types = ["L", "t1i", "t1f", "t2i", "t2f", "Mmass", "mx"]
        for i in range(tree.ncomp):
            compstr = f"0{i}" if i < 10 else f"{i}"
            hdl.write(f"comp-{compstr}: ")
            for j in range(7):
                hdl.write(f"{dv_types[j]} {xarr[7*i+j]:.3e}, ")
            hdl.write("\n")
        hdl.close()

        # RMS frequency errors?
        freqs4 = freqs[:4]
        freq_err = np.linalg.norm(freqs4 - target_eigvals) / np.linalg.norm(target_eigvals)
        beam3d.freq_err_hist += [freq_err]

        beam3d.freq_hist += [list(freqs4)]
        # print(f"{beam3d.freq_hist=}")
        # exit()

        return funcs, False

    def get_function_sens(x_dict, funcs):
        xlist = x_dict["vars"]
        xarr = np.array([float(_) for _ in xlist])

        sens = {}

        funcs = {}
        funcs['dummyobj'] = 0.0

        if args.freq:
            for imode in range(nmodes):
                freq_grad = beam3d.get_frequency_gradient(xarr, imode)
                sens[f'freq{imode}'] = {'vars': freq_grad}

        mass_gradient = beam3d.get_mass_gradient(xarr)
        sens['dummyobj'] = {'vars' : 0.0 * mass_gradient}
        if args.mass:
            sens['mass'] = {'vars': mass_gradient} # add actual later
        
        if args.cg:
            centroid_Grad = tree.get_centroid_gradient(xarr)
            for i in range(3):
                sens[f'centroid{i}'] = {'vars' : centroid_Grad[i,:] }

        if args.failure:
            fail_index_grad = beam3d.get_failure_index_gradient(xarr)
            sens['failure'] = { 'vars' : fail_index_grad }

        return sens, False
    
    opt_problem = Optimization("tuning-fork", get_functions)
    opt_problem.addVarGroup(
        "vars",
        num_dvs,
        lower=np.array(([1e-2] + [1e-4]*4 + [0.1,0.1])*ncomp), # TODO : change min of non-struct-masses?
        upper=np.array(([10.0] + [1e0]*4 + [30.0,0.9])*ncomp),
        value=init_design,
        scale=np.array(([1.0] + [1e2]*4 + [1e-1,1e0])*ncomp),
    )

    # note - may be better to change it to a frequency error objective later
    # we'll see..

    # how are you supposed to scale it again
    opt_problem.addObj('dummyobj')
    if args.mass:
        opt_problem.addCon("mass", scale=1.0/target_mass, lower=target_mass, upper=target_mass) #scale=1.0/target_mass**2)
    if args.freq:
        for imode in range(nmodes):
            opt_problem.addCon(
                f"freq{imode}",
                lower=target_eigvals[imode],
                upper=target_eigvals[imode],
                scale=1.0/target_eigvals[imode]
            )
    if args.cg:
        for i in range(3):
            opt_problem.addCon(f'centroid{i}', scale=1.0/target_centroid[i],
                               lower=target_centroid[i],
                               upper=target_centroid[i])
    
    if args.failure:
        opt_problem.addCon('failure', scale=1.0, upper=1.0)

    # run an SNOPT optimization
    snoptimizer = SNOPT(
        options={
            "Print frequency": 1000,
            "Summary frequency": 10000000,
            "Major feasibility tolerance": 1e-6,
            "Major optimality tolerance": 1e-4,
            "Verify level": -1,
            "Major iterations limit": 1000, #1000, # 1000,
            "Minor iterations limit": 150000000,
            "Iterations limit": 100000000,
            "Major step limit": 5e-2,
            "Nonderivative linesearch": None,
            "Linesearch tolerance": 0.9,
            #"Difference interval": 1e-6,
            "Function precision": 1e-6, #results in btw 1e-4, 1e-6 step sizes
            "New superbasics limit": 2000,
            "Penalty parameter": 1,
            "Scale option": 1,
            "Hessian updates": 40,
            "Print file": os.path.join(args.output, "SNOPT_print.out"),
            "Summary file": os.path.join(args.output, "SNOPT_summary.out"),
        }
    )

    hot_start = False
    sol = snoptimizer(
        opt_problem,
        sens=get_function_sens,
        storeHistory="myhist.hst",
        hotStart="myhist.hst" if hot_start else None,
    )

    # print final solution
    sol_xdict = sol.xStar
    print(f"Final solution = {sol_xdict}", flush=True)

    # print the final design
    xopt_list = sol_xdict["vars"]
    xopt_arr = np.array([float(_) for _ in xopt_list])
    xopt_lengths = xopt_arr[0::3]
    xopt_xpts = tree.get_xpts(xopt_lengths)
    tree.plot_mesh(xopt_xpts, filename=f"{args.output}/opt-design-mesh.png")

    # plot frequency error list
    # ---------------------------------
    freq_err_hist = beam3d.freq_err_hist
    iterations = [_ for _ in range(len(freq_err_hist))]

    import matplotlib.pyplot as plt
    import niceplots
    plt.close('all')
    plt.style.use(niceplots.get_style())
    plt.figure()
    plt.plot(iterations, freq_err_hist, 'k')
    plt.margins(x=0.05, y=0.05)
    plt.xlabel("Iterations")
    plt.ylabel("Freq Error")
    plt.yscale('log')
    plt.savefig(f"{args.output}/freq-err-hist.png", dpi=400)

    freq_hist = beam3d.freq_hist
    plt.close('all')
    plt.style.use(niceplots.get_style())
    plt.figure(figsize=(10,6))
    # colors = plt.cm.Cu(np.linspace(0.0, 1.0, 4))
    # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colors = plt.cm.plasma(np.linspace(0.0, 1.0, 4))

    for i in [3,2,1,0]:
        _this_freq_hist = [_[i] for _ in freq_hist]
        target_i = [target_eigvals[i] for _ in range(len(iterations))]
        plt.plot(iterations, target_i, '--', color=colors[i])
        plt.plot(iterations, _this_freq_hist, color=colors[i], label=f"freq{i}")
        
    plt.margins(x=0.05, y=0.05)
    plt.xlabel("Iterations")
    plt.ylabel("Frequencies")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1)) 
    plt.yscale('log')
    plt.savefig(f"{args.output}/freq-hist.png", dpi=400)