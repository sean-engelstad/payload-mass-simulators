import sys, numpy as np, os, time
from payload_mass_sim import *
from pyoptsparse import SNOPT, Optimization

if not os.path.exists("_modal"):
    os.mkdir("_modal")

if __name__ == "__main__":

    material = Material.aluminum()

    # test out the tree data structure
    # TODO : modify tree start nodes and directions graph
    #    in order to have level 2 optimization
    tree = TreeData(
        tree_start_nodes=[0] + [1]*4 + [2,3,4,5],
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

    beam3d = BeamAssembler(material, tree)

    demo = False
    if demo:
        # now build 3D beam solver and solve the eigenvalue problem
        freqs = beam3d.get_frequencies(init_design)
        print(f"{freqs=}")
        nmodes = 5
        beam3d.plot_eigenmodes(
            nmodes=nmodes,
            show=False,
            def_scale=0.5
        )

        # see if we can get derivatives of the natural frequencies now
        freq_grads = np.zeros((num_dvs, nmodes))
        start_time = time.time()
        print("Getting eigenvalue derivatives:")
        for imode in range(5):
            freq_grads[:,imode] = beam3d.get_frequency_gradient(init_design, imode)
        dt = time.time() - start_time
        print(f"\tcomputed freq grads in {dt:.4f} seconds.") 

    debug = False
    if debug:
        # FD test on the gradients
        for imode in range(4):
            beam3d.freq_FD_test(init_design, imode, h=1e-5)
            beam3d.dKdx_FD_test(init_design, imode, h=1e-5)
            beam3d.dMdx_FD_test(init_design, imode, h=1e-5)
        exit()

    # optimization
    # -------------------------------------

    nmodes = 4 # we're targeting the first 4 eigenvalues for now

    target_eigvals = np.array([24.3, 29.4, 99.5, 173.6])
    target_mass = 176.407 # kg
    target_centroid = np.array([0.0104782366, 0.005086985, 0.417962588]) #m

    def get_functions(x_dict):
        xlist = x_dict["vars"]
        xarr = np.array([float(_) for _ in xlist])

        freqs = beam3d.get_frequencies(xarr)

        funcs = {
            f'freq{imode}':freqs[imode] for imode in range(nmodes)
        }
        mass = beam3d.get_mass(xarr)
        funcs['mass'] = (mass - target_mass)**2

        centroid = tree.get_centroid(xarr)
        centroidError = (centroid - target_centroid)**2
        # for i in range(3):
            # funcs[f'centroid{i}'] = centroidError[i]

        # writeout a current opt-status.txt file
        hdl = open("opt-status.txt", mode='w')
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
            else:
                hdl.write(f"\tfunc {key} = {funcs[key]:.4e}\n")
        hdl.write("vars:\n")
        dv_types = ["L", "t1", "t2"]
        for i in range(num_dvs):
            dv_type = dv_types[i%3]
            icomp = i // 3
            dv_name = f"{dv_type}-comp{icomp}"
            hdl.write(f"\tvar {dv_name} {xarr[i]:.4e}\n")
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
        for imode in range(nmodes):
            freq_grad = beam3d.get_frequency_gradient(xarr, imode)
            sens[f'freq{imode}'] = {'vars': freq_grad}

        mass_gradient = beam3d.get_mass_gradient(xarr)
        sens['mass'] = {'vars': 2 * (beam3d.get_mass(xarr) - target_mass) * mass_gradient} # add actual later
        
        centroid_gradient = tree.get_centroid_gradient(xarr)
        # for i in range(3):
            # sens[f'centroid{i}'] = {'vars': 2 * (tree.get_centroid(xarr) - target_centroid)[i] * centroid_gradient[i,:]}

        return sens, False
    
    opt_problem = Optimization("tuning-fork", get_functions)
    opt_problem.addVarGroup(
        "vars",
        num_dvs,
        lower=np.array([0.01,1e-3,1e-3]*ncomp),
        upper=np.array([3.0,1e0,1e0]*ncomp),
        value=np.array([0.3,5e-3,5e-3]*ncomp),
        scale=np.array([1.0,1e2,1e2]*ncomp),
    )

    # note - may be better to change it to a frequency error objective later
    # we'll see..

    opt_problem.addObj("mass", scale=target_mass**2)
    for imode in range(nmodes):
        opt_problem.addCon(
            f"freq{imode}",
            lower=target_eigvals[imode],
            upper=target_eigvals[imode],
            scale=1e-2
        )

    # for i in range(3):  # Constraints for x, y, and z centroid coordinates
    #     opt_problem.addCon(
    #         f"centroid{i}",
    #         lower=target_centroid[i],
    #         upper=target_centroid[i],
    #         scale=1e-2
    #     )

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
            "Print file": os.path.join("SNOPT_print.out"),
            "Summary file": os.path.join("SNOPT_summary.out"),
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
    tree.plot_mesh(xopt_xpts, filename="opt-design-mesh.png")

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
    plt.savefig("_modal/freq-err-hist.png", dpi=400)

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
    plt.savefig("_modal/freq-hist.png", dpi=400)