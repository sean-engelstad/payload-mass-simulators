from tacs import caps2tacs
from mpi4py import MPI
from funtofem import *
from pyoptsparse import SNOPT, Optimization
from tacs.functions import *
import os

# run a steady elastic structural analysis in TACS using the tacsAIM wrapper caps2tacs submodule
# -------------------------------------------------------------------------------------------------
# 1: build the tacs aim, egads aim wrapper classes

comm = MPI.COMM_WORLD
f2f_model = FUNtoFEMmodel("fork")
fork = Body.aeroelastic(
    "fork"
)

print(f"proc on rank {comm.rank}")

tacs_model = caps2tacs.TacsModel.build(
    csm_file="tuning-fork-small.csm", comm=comm, active_procs=[0]
)
tacs_model.mesh_aim.set_mesh(
    edge_pt_min=5,
    edge_pt_max=20,
    global_mesh_size=1.0,
    max_surf_offset=0.01,
    max_dihedral_angle=15,
).register_to(tacs_model)
tacs_aim = tacs_model.tacs_aim

aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

# nstiff = int(tacs_model.get_config_parameter("stiffener_count"))
# nskin = nstiff+1

# capsGroups - 
init_thickness = 0.05
thick_scale = 10 # was 100, prob larger here?


for level in range(1, 1+1):
    for face in ['xp', 'xn', 'zp', 'zn']:
        for orientation in ['', 'v']:
            capsGroup = f"{face}{level}{orientation}T"
            caps2tacs.ShellProperty(
                caps_group=capsGroup, material=aluminum, membrane_thickness=init_thickness
            ).register_to(tacs_model)
            Variable.structural(capsGroup, value=init_thickness).set_bounds(
                lower=1e-4, upper=1.0, scale=thick_scale
            ).register_to(fork)

capsGroup = "baseT"
caps2tacs.ShellProperty(
    caps_group=capsGroup, material=aluminum, membrane_thickness=init_thickness
).register_to(tacs_model)
Variable.structural(capsGroup, value=init_thickness).set_bounds(
    lower=1e-4, upper=1.0, scale=thick_scale
).register_to(fork)

caps2tacs.PinConstraint("root",dof_constraint=12346).register_to(tacs_model)
#caps2tacs.GridForce("loading", direction=[0, 0, 1.0], magnitude=10).register_to(tacs_model)

f2f_model.structural = tacs_model

# SHAPE VARIABLES
shape_var_list = [2, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
ct = 0
for dim in ["l", "w"]:
    for level in [""]: #["","2"]:
        for dir in ["x","z"]:
            Variable.shape(f"{dir}lat{level}:{dim}", value=shape_var_list[ct]).set_bounds(
                lower=0.1, upper=10.0
            ).register_to(fork)
            ct += 1

# shape_var_list = [4, 4, 2, 2, 0.5, 0.5, 0.5, 0.5]
# ct = 0        
# for dim in ["h", "w"]:
#     for level in ["","2"]:
#         for dir in ["x","z"]:
#             Variable.shape(f"{dir}{level}:{dim}", value=shape_var_list[ct]).set_bounds(
#                 lower=0.1, upper=10.0
#             ).register_to(fork)
#             ct += 1

# shape_var_list = [1, 3]
# ct = 0
# for dim in ["w", "h"]:
#     Variable.shape(f"base:{dim}", value=1.0).set_bounds(
#         lower=0.1, upper=10.0
#     ).register_to(fork)
#     ct += 1

# register the funtofem Body to the model
fork.register_to(f2f_model)

# add analysis functions to the model
# caps2tacs.AnalysisFunction.ksfailure(ksWeight=50.0, safetyFactor=1.5).register_to(
#     tacs_model
# )
# caps2tacs.AnalysisFunction.mass().register_to(tacs_model)

# make the scenario(s)
tacs_scenario = Scenario.steady("tacs", steps=1)
Function.mass().optimize(scale=1.0e-2, objective=True, plot=True).register_to(
    tacs_scenario
)
# Function.ksfailure(ks_weight=10.0).optimize(
#     scale=30.0, upper=0.267, objective=False, plot=True
# ).register_to(tacs_scenario)
tacs_scenario.register_to(f2f_model)

# #Linear constraints
# # ---------------------
# left_var = f2f_model.get_variables(f"{comp_group}{icomp}-{adj_type}")
# right_var = f2f_model.get_variables(f"{comp_group}{icomp+1}-{adj_type}")
# # print(f"left var = {left_var}, right var = {right_var}")
# adj_constr = left_var - right_var
# adj_constr.set_name(f"{comp_group}{icomp}-adj_{adj_type}").optimize(
#     lower=-adj_value, upper=adj_value, scale=10.0, objective=False
# ).register_to(f2f_model)



# run the pre analysis to build tacs input files
# alternative is to call tacs_aim.setup_aim().pre_analysis() with tacs_aim = tacs_model.tacs_aim
# tacs_model.setup(include_aim=True)
tacs_aim.setup_aim()
# tacs_aim.pre_analysis()

comm.Barrier()

# MODAL analysis
# ------------------
target_eigvals = [24.3, 29.4, 99.5, 173.6, 0.0]
class TacsModalShape:
    def __init__(self, tacs_model, f2f_model, sigma=10.0, nEig=6):
        self.tacs_model = tacs_model
        self.tacs_aim = tacs_model.tacs_aim
        self.f2f_model = f2f_model
        self.sigma = sigma # eigenvalue guess
        self.nEig = nEig

    def get_functions(self, x_dict):
        # need to update the variables here 
        # and set them into tacs aim somehow.. through FUNtoFEM and tacs_aim
        for var in self.f2f_model.get_variables():
            for var_key in x_dict:
                if var.name == var_key:
                    # assumes here that only pyoptsparse single variables (no var groups are made)
                    var.value = float(x_dict[var_key])

        # update the tacs model design
        input_dict = {var.name: var.value for var in self.f2f_model.get_variables()}
        self.f2f_model.structural.update_design(input_dict)

        self.tacs_aim.pre_analysis()

        # fea_solver is the pytacs object
        fea_solver = self.tacs_model.fea_solver(0)
        fea_solver.initialize(tacs_model._callback)

        if fea_solver.bdfInfo.is_xrefed is False:
            fea_solver.bdfInfo.cross_reference()
            fea_solver.bdfInfo.is_xrefed = True

        # create the modal problem
        sigma = self.sigma; nEigs = self.nEig
        MP = fea_solver.createModalProblem("fork-modal-problem", sigma, nEigs)

        tacs_funcs = {}
        MP.solve()
        MP.evalFunctions(tacs_funcs)
        # MP.evalFunctionsSens(tacs_sens, evalFuncs=function_names)
        MP.writeSolution(
            baseName="modal", outputDir=tacs_model.analysis_dir(proc=0)
        )

        # write a dummy sens file (dummy because evalFuncs is None)
        MP.writeSensFile(evalFuncs=None, tacsAim=self.tacs_aim)

        self.tacs_aim.post_analysis()

        funcs = {}
        if comm.rank == 0:
            func_keys = list(tacs_funcs.keys())
            func_names = [f'eigsm.{i}' for i in range(nEigs)]
            funcs = {func_names[i]:tacs_funcs[func_keys[i]] for i in range(len(func_keys))}
        else:
            funcs = None
        funcs = comm.bcast(funcs, root=0)

        # static problem
        # ---------------------
        # SPs = tacs_model.createTACSProbs(addFunctions=True)

        # # solve each structural analysis problem (in this case 1)
        # tacs_funcs2 = {}
        # # tacs_sens = {}
        # for caseID in SPs:
        #     SPs[caseID].addFunction(funcName="mass", funcHandle=StructuralMass)
        #     SPs[caseID].solve()
        #     SPs[caseID].evalFunctions(tacs_funcs2, evalFuncs=['mass'])
        #     # SPs[caseID].evalFunctionsSens(tacs_sens, evalFuncs=['mass'])
        #     SPs[caseID].writeSolution(
        #         baseName="struct", outputDir=tacs_model.analysis_dir(proc=0)
        #     )
        
        # keys2 = list(tacs_funcs2)
        # # print(f"{tacs_funcs2=}")
        # funcs['mass-err'] = (tacs_funcs2[keys2[0]] - 0.00571014)**2 # kg error

        # just empty objective for now
        funcs['mass-err'] = 0.0

        # writeout the new design and funcs to a file
        hdl = open("opt-status.txt", mode='w')
        hdl.write("funcs:\n")
        funcs_keys = list(funcs.keys())
        for i,key in enumerate(funcs_keys):
            if "eig" in key:
                hdl.write(f"\tfunc {key} = {funcs[key]:.4e}, target {target_eigvals[i]}\n")
            else:
                hdl.write(f"\tfunc {key} = {funcs[key]:.4e}")
        hdl.write("vars:\n")
        for var in f2f_model.get_variables():
            hdl.write(f"\tvar {var.name} = {var.value:.4e}\n")
        hdl.close()

        fail = False
        return funcs, fail
    
    def get_function_sens(self, x_dict, funcs):
        # need to update the variables here 
        # and set them into tacs aim somehow.. through FUNtoFEM and tacs_aim
        for var in self.f2f_model.get_variables():
            for var_key in x_dict:
                if var.name == var_key:
                    # assumes here that only pyoptsparse single variables (no var groups are made)
                    var.value = float(x_dict[var_key])

        # update the tacs model design
        input_dict = {var.name: var.value for var in self.f2f_model.get_variables()}
        self.f2f_model.structural.update_design(input_dict)

        self.tacs_aim.pre_analysis()

        # fea_solver is the pytacs object
        fea_solver = self.tacs_model.fea_solver(0)
        fea_solver.initialize(tacs_model._callback)

        if fea_solver.bdfInfo.is_xrefed is False:
            fea_solver.bdfInfo.cross_reference()
            fea_solver.bdfInfo.is_xrefed = True

        # create the modal problem
        sigma = self.sigma; nEigs = self.nEig
        MP = fea_solver.createModalProblem("fork-modal-problem", sigma, nEigs)

        # print(f"checkpt1")

        # TODO : could avoid running the analysis all over again
        # use gatekeeper function like Marshall did (optional though since very cheap here)
        tacs_funcs = {}
        tacs_sens = {}
        MP.solve()
        # print(f"checkpt2")
        MP.evalFunctions(tacs_funcs)
        print(f"pre-modal-adjoint")
        MP.evalFunctionsSens(tacs_sens)
        print(f"post-modal-adjoint")
        MP.writeSolution(
            baseName="modal", outputDir=tacs_model.analysis_dir(proc=0)
        )
        # print(f"checkpt4")

        # write a dummy sens file (dummy because evalFuncs is None)
        evalFuncs = [f'eigsm.{i}' for i in range(nEigs)]
        print(f"{evalFuncs=}")
        MP.writeSensFile(evalFuncs=evalFuncs, tacsAim=self.tacs_aim)

        # print(f"checkpt5")

        self.tacs_aim.post_analysis()

        # now get the shape derivatives here..
        # and add thickness derivatives to a dict here
        sens = {}
        if comm.rank == 0: # may need to broadcast from root?
            funckeys = tacs_funcs.keys()
            for ifunc,funckey in enumerate(funckeys):
                func_name = evalFuncs[ifunc]
                sens[func_name] = {}
                for ivar,var in enumerate(self.f2f_model.get_variables()):
                    if var.analysis_type == "structural":
                        sens[func_name][var.name] = tacs_sens[funckey]['struct'][ivar]
                    elif var.analysis_type == "shape":
                        
                        sens[func_name][var.name] = self.tacs_aim.aim.dynout[
                            func_name
                        ].deriv(var.name)
        else:
            sens = None
        sens = comm.bcast(sens, root=0)

        # static problem
        # ---------------------
        # SPs = tacs_model.createTACSProbs(addFunctions=True)

        # # solve each structural analysis problem (in this case 1)
        # # tacs_funcs2 = {}
        # tacs_sens2 = {}
        # for caseID in SPs:
        #     SPs[caseID].addFunction(funcName="mass", funcHandle=StructuralMass)
        #     SPs[caseID].solve()
        #     # SPs[caseID].evalFunctions(tacs_funcs2, evalFuncs=['mass'])
        #     SPs[caseID].evalFunctionsSens(tacs_sens2, evalFuncs=['mass'])
        #     SPs[caseID].writeSolution(
        #         baseName="struct", outputDir=tacs_model.analysis_dir(proc=0)
        #     )
        
        # keys2 = list(tacs_sens2)
        # # print(f"{tacs_sens2=}")
        # funcs['mass-err'] = (tacs_sens2[keys2[0]] - 0.00571014)**2 # kg error

        # just empty objective gradient for now
        sens['mass-err'] = {}
        for ivar,var in enumerate(self.f2f_model.get_variables()):
            sens['mass-err'][var.name] = 0.0

        fail = False
        return sens, fail

# demo the forward analysis
# can lower nEig to 3 to make it run faster..
# would like nEig = 6 later
tacs_modal_shape = TacsModalShape(tacs_model, f2f_model, sigma=10.0, nEig=4)

# debugging
# funcs,_ = tacs_modal_shape.get_functions({}); print(f"{funcs=}")
# sens,_ = tacs_modal_shape.get_function_sens({}, None); print(f"{sens=}")
# exit()

# function names for eigenvalues are "fork-model-problem_eigsm.i" for i=0,...,5
# now we need to setup pyoptsparse optimizer..


# create the pyoptsparse optimization problem
opt_problem = Optimization("gbm-AE-sizing", tacs_modal_shape.get_functions)

for var in f2f_model.get_variables():
    opt_problem.addVar(
        var.name,
        lower=var.lower,
        upper=var.upper,
        value=var.value,
        scale=var.scale
    )

# mass constraint should be:
# mass = 0.005710147
opt_problem.addObj(
    f"mass-err",
    scale=1e0,
)

# TODO : fix mass-err to not be 0
# TODO : add CG, 2 more eigvals (first 6), and later the modal mass constraints
#    using eigenvectors
# TODO : survive inertial loads, check with failure index of linear static analysis

for ieig in range(tacs_modal_shape.nEig):
    opt_problem.addCon(
        f"eigsm.{ieig}",
        lower=target_eigvals[ieig],
        upper=target_eigvals[ieig],
        scale=1e-2,
    )

# add funtofem model variables to pyoptsparse
# manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-4,
        "Verify level": -1,
        "Major iterations limit": 1000,
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
    sens=tacs_modal_shape.get_function_sens,
    storeHistory="myhist.hst",
    hotStart="myhist.hst" if hot_start else None,
)

# print final solution
sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)

# target mass objective too?
# setup eigenvalue constraints..
# probably need some way to also do integer design variables..