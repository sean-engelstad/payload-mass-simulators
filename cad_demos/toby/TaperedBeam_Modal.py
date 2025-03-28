from tacs import caps2tacs
from mpi4py import MPI

# run a steady elastic structural analysis in TACS using the tacsAIM wrapper caps2tacs submodule
# -------------------------------------------------------------------------------------------------
# 1: build the tacs aim, egads aim wrapper classes

comm = MPI.COMM_WORLD

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

aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

caps2tacs.ThicknessVariable(
    caps_group="all", value=0.05, material=aluminum
).register_to(tacs_model)

# # SHAPE VAR FOR DEBUGGING, CAN REMOVE THIS
# caps2tacs.ShapeVariable("rib_a1", value=1.0).register_to(tacs_model)

# add constraints and loads
caps2tacs.PinConstraint("base").register_to(tacs_model)
caps2tacs.GridForce("loading", direction=[0, 0, 1.0], magnitude=100).register_to(tacs_model)

# add analysis functions to the model
caps2tacs.AnalysisFunction.ksfailure(ksWeight=50.0, safetyFactor=1.5).register_to(
    tacs_model
)
caps2tacs.AnalysisFunction.mass().register_to(tacs_model)

# run the pre analysis to build tacs input files
# alternative is to call tacs_aim.setup_aim().pre_analysis() with tacs_aim = tacs_model.tacs_aim
tacs_model.setup(include_aim=True)

comm.Barrier()

# ----------------------------------------------------------------------------------
# 2. Run the TACS steady elastic structural analysis, forward + adjoint

# choose method 1 or 2 to demonstrate the analysis : 1 - fewer lines, 2 - more lines
# method = 1

# show both ways of performing the structural analysis in more or less detail
# if method == 1:  # less detail version
# print(f"tacs - preanalysis rank {comm.rank}", flush=True)
# tacs_model.pre_analysis()
# print(f"tacs - run analysis rank {comm.rank}", flush=True)
# tacs_model.run_analysis()
# print(f"tacs - post analysis rank {comm.rank}", flush=True)
# tacs_model.post_analysis()

# print("Tacs model static analysis outputs...\n")
# for func in tacs_model.analysis_functions:
#     print(f"func {func.name} = {func.value}")

#     for var in tacs_model.variables:
#         derivative = func.get_derivative(var)
#         print(f"\td{func.name}/d{var.name} = {derivative}")
#     print("\n")

# fea_solver is the pytacs object
fea_solver = tacs_model.fea_solver(0)
fea_solver.initialize(tacs_model._callback)

if fea_solver.bdfInfo.is_xrefed is False:
    fea_solver.bdfInfo.cross_reference()
    fea_solver.bdfInfo.is_xrefed = True

# create the modal problem
sigma = 10.0 # eigenvalue guess, use closed-form solution
nEigs = 3
MP = fea_solver.createModalProblem("beam-modal-problem", sigma, nEigs)

MP.solve()
# MP.evalFunctions(tacs_funcs, evalFuncs=function_names)
# MP.evalFunctionsSens(tacs_sens, evalFuncs=function_names)
MP.writeSolution(
    baseName="tacs_output", outputDir=tacs_model.analysis_dir(proc=0)
)

# this creates linear static
# SPs = fea_solver.createTACSProbsFromBDF()



# elif method == 2:  # longer way that directly uses pyTACS & static problem routines
# SPs = tacs_model.createTACSProbs(addFunctions=True)

# # solve each structural analysis problem (in this case 1)
# tacs_funcs = {}
# tacs_sens = {}
# function_names = tacs_model.function_names
# for caseID in SPs:
#     SPs[caseID].solve()
#     SPs[caseID].evalFunctions(tacs_funcs, evalFuncs=function_names)
#     SPs[caseID].evalFunctionsSens(tacs_sens, evalFuncs=function_names)
#     SPs[caseID].writeSolution(
#         baseName="tacs_output", outputDir=tacs_model.analysis_dir(proc=0)
#     )

# # print the output analysis functions and sensitivities
# print("\nTacs Analysis Outputs...")
# for func_name in function_names:
#     # find the associated tacs key (tacs key = (loadset) + (func_name))
#     for tacs_key in tacs_funcs:
#         if func_name in tacs_key:
#             break

#     func_value = tacs_funcs[tacs_key].real
#     print(f"\tfunctional {func_name} = {func_value}")

#     struct_sens = tacs_sens[tacs_key]["struct"]
#     print(f"\t{func_name} structDV gradient = {struct_sens}")

#     xpts_sens = tacs_sens[tacs_key]["Xpts"]
#     print(f"\t{func_name} coordinate derivatives = {xpts_sens}")

#     print("\n")
