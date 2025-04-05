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
f2f_model = FUNtoFEMmodel("beam")
beam = Body.aeroelastic(
    "beam"
)

print(f"proc on rank {comm.rank}")

tacs_model = caps2tacs.TacsModel.build(
    csm_file="taperedBeam.csm", comm=comm, active_procs=[0]
)
tacs_model.mesh_aim.set_mesh(
    edge_pt_min=15,
    edge_pt_max=20,
    global_mesh_size=0.25,
    max_surf_offset=0.01,
    max_dihedral_angle=15,
).register_to(tacs_model)
tacs_aim = tacs_model.tacs_aim

aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nstiff = int(tacs_model.get_config_parameter("stiffener_count"))
nskin = nstiff+1

# capsGroups - 
init_thickness = 0.05
thick_scale = 10 # was 100, prob larger here?

#Apply properties to base and endmass?
for capsGroup in ["base", "endmass"]: #"root", "tip"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(capsGroup, value=init_thickness).set_bounds(
        lower=1e-4, upper=1.0, scale=thick_scale
    ).register_to(beam)

#apply initial thickness to all skins
for iskin in range(1, nskin+1):
    capsGroup = f"skin{iskin}"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(capsGroup, value=init_thickness).set_bounds(
        lower=1e-4, upper=1.0, scale=thick_scale
    ).register_to(beam)

#apply initial thickness to all stiffeners
for istiff in range(1, nstiff+1):
    capsGroup = f"stiff{istiff}"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(capsGroup, value=init_thickness).set_bounds(
        lower=1e-4, upper=1.0, scale=thick_scale
    ).register_to(beam)
    # New 02/28 - independent stiffener positions and hole fractions
    #stiffener position
    Variable.shape(f"stiff{istiff}_relative_pos", value=0.1).set_bounds(
        lower=0.001, upper=0.999
    ).register_to(beam)

    #hole fraction
    Variable.shape(f"stiff{istiff}_holefrac", value=0.5).set_bounds(
        lower=0.001, upper=0.999
    ).register_to(beam)

# New 02/28 - independent skin thickness for all 4 sides in each gap between stiffeners
for gap in range(1, nstiff):
    for side in ["xy_front", "xy_back", "yz_front", "yz_back"]:
        Variable.structural(f"gap_{nstiff}_skin_{side}_thickness", value=0.1).set_bounds(
            lower=0.01, upper=0.5
        ).register_to(beam)

caps2tacs.PinConstraint("base").register_to(tacs_model)
caps2tacs.GridForce("loading", direction=[0, 0, 1.0], magnitude=10).register_to(tacs_model)

f2f_model.structural = tacs_model

# SHAPE VARIABLES
for direc in ["x", "z"]:
    Variable.shape(f"beam:root_d{direc}", value=1.0).set_bounds(
        lower=0.1, upper=10.0
    ).register_to(beam)
    Variable.shape(f"beam:taper_{direc}", value=0.5).set_bounds(
        lower=0.01, upper=1.5
    ).register_to(beam)
Variable.shape(f"beam:length", value=2.0).set_bounds(
    lower=0.1, upper=10.0
).register_to(beam)

for direc in ["dx", "dy", "dz"]:
    Variable.shape(f"endmass:{direc}", value=1.0).set_bounds(
        lower=0.1, upper=3.0
    ).register_to(beam)

for mystr in ["poly", "hole_poly"]:
    Variable.shape(f"stiffener:{mystr}B", value=1.0).set_bounds(
        lower=0.6, upper=1.4
    ).register_to(beam)
    Variable.shape(f"stiffener:{mystr}C", value=0.0).set_bounds(
        lower=-0.3, upper=0.3
    ).register_to(beam)

# register the funtofem Body to the model
beam.register_to(f2f_model)

# add analysis functions to the model
# caps2tacs.AnalysisFunction.ksfailure(ksWeight=50.0, safetyFactor=1.5).register_to(
#     tacs_model
# )
# caps2tacs.AnalysisFunction.mass().register_to(tacs_model)

# make the scenario(s)
tacs_scenario = Scenario.steady("tacs", steps=100)
Function.mass().optimize(scale=1.0e-2, objective=True, plot=True).register_to(
    tacs_scenario
)
# Function.ksfailure(ks_weight=10.0).optimize(
#     scale=30.0, upper=0.267, objective=False, plot=True
# ).register_to(tacs_scenario)
tacs_scenario.register_to(f2f_model)

# run the pre analysis to build tacs input files
# alternative is to call tacs_aim.setup_aim().pre_analysis() with tacs_aim = tacs_model.tacs_aim
# tacs_model.setup(include_aim=True)
tacs_aim.setup_aim()
# tacs_aim.pre_analysis()

comm.Barrier()

# MODAL analysis
# ------------------

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
        MP = fea_solver.createModalProblem("beam-modal-problem", sigma, nEigs)

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
        target_eigvals = [24.4, 29.9, 42.3, 45.2, 0]
        hdl = open("opt-status.txt", mode='w')
        hdl.write("funcs:\n")
        funcs_keys = list(funcs.keys())
        for i,key in enumerate(funcs_keys):
            hdl.write(f"\tfunc {key} = {funcs[key]:.4e}, target {target_eigvals[i]}\n")
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
        MP = fea_solver.createModalProblem("beam-modal-problem", sigma, nEigs)

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

# function names for eigenvalues are "beam-model-problem_eigsm.i" for i=0,...,5
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

#NEW 02/28: add constraint so that stiffeners do not overlap
for istiff in range(2, nstiff + 1):  # begin from the second stiffener
    opt_problem.addCon(
        f"stiff{istiff}_order",
        lower=0.01,  # Minimum separation
        upper=None,
        linear=True,
        wrt=[f"stiff{istiff}_relative_pos", f"stiff{istiff-1}_relative_pos"],
        jac={f"stiff{istiff}_relative_pos": 1, f"stiff{istiff-1}_relative_pos": -1}, #Syntax???
    )  

# TODO : fix mass-err to not be 0
# TODO : add CG, 2 more eigvals, and later the modal mass constraints
#    using eigenvectors

target_eigvals = [24.4, 29.9, 42.3, 45.2]
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