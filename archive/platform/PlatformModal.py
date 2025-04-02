from tacs import caps2tacs
from mpi4py import MPI
#from funtofem import *

# run a steady elastic structural analysis in TACS using the tacsAIM wrapper caps2tacs submodule
# -------------------------------------------------------------------------------------------------
# 1: build the tacs aim, egads aim wrapper classes

comm = MPI.COMM_WORLD

print(f"proc on rank {comm.rank}")

tacs_model = caps2tacs.TacsModel.build(
    csm_file="PlatformStructure.csm", comm=comm, active_procs=[0]
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



# capsGroups - 
# Assign shell properties to column structures
for i in range(1, 5):  # 4 columns
    capsGroup = f"column{i}"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=0.01
    ).register_to(tacs_model)

# Assign shell properties to base plates
for i in range(1, 5):  # 4 base plates
    capsGroup = f"base{i}"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=0.02
    ).register_to(tacs_model)

# Assign shell properties to floors (num_floors total)
num_floors = int(tacs_model.get_config_parameter("num_floors"))
for i in range(1, num_floors + 1):  
    capsGroup = f"floor{i}"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=0.05
    ).register_to(tacs_model)



# Update the constraints and loads
# Apply pin constraints to all base plates (to fix them in place)
#for i in range(1, 5):  
caps2tacs.PinConstraint("base").register_to(tacs_model)

# Apply a uniform grid force to the structure
caps2tacs.GridForce("loading", direction=[0, 0, 1.0], magnitude=10).register_to(tacs_model)



# #f2f_model.structural = tacs_model

# # SHAPE VARIABLES
# # Shape Variables for Column Dimensions
# Variable.shape("column:width", value=0.1).set_bounds(
#     lower=0.05, upper=0.3
# ).register_to(tacs_model)

# Variable.shape("column:height", value=5.0).set_bounds(
#     lower=1.0, upper=10.0
# ).register_to(tacs_model)

# # Shape Variables for Floor Size
# Variable.shape("floor:length", value=4.0).set_bounds(
#     lower=2.0, upper=10.0
# ).register_to(tacs_model)

# Variable.shape("floor:width", value=2.0).set_bounds(
#     lower=1.0, upper=5.0
# ).register_to(tacs_model)

# # Shape Variables for Base Plates
# Variable.shape("base:length", value=1.0).set_bounds(
#     lower=0.5, upper=3.0
# ).register_to(tacs_model)

# Variable.shape("base:width", value=1.0).set_bounds(
#     lower=0.5, upper=3.0
# ).register_to(tacs_model)



# # register the funtofem Body to the model
# wing.register_to(f2f_model)



# add analysis functions to the model
caps2tacs.AnalysisFunction.ksfailure(ksWeight=50.0, safetyFactor=1.5).register_to(
    tacs_model
)
caps2tacs.AnalysisFunction.mass().register_to(tacs_model)



# # make the scenario(s)
# tacs_scenario = Scenario.steady("tacs", steps=100)
# Function.mass().optimize(scale=1.0e-2, objective=True, plot=True).register_to(
#     tacs_scenario
# )
# # Function.ksfailure(ks_weight=10.0).optimize(
# #     scale=30.0, upper=0.267, objective=False, plot=True
# # ).register_to(tacs_scenario)
# tacs_scenario.register_to(f2f_model)



# run the pre analysis to build tacs input files
# alternative is to call tacs_aim.setup_aim().pre_analysis() with tacs_aim = tacs_model.tacs_aim
# tacs_model.setup(include_aim=True)
tacs_aim.setup_aim()
tacs_aim.pre_analysis()

comm.Barrier()

# MODAL analysis
# ------------------

# fea_solver is the pytacs object
fea_solver = tacs_model.fea_solver(0)
fea_solver.initialize(tacs_model._callback)

if fea_solver.bdfInfo.is_xrefed is False:
    fea_solver.bdfInfo.cross_reference()
    fea_solver.bdfInfo.is_xrefed = True

# create the modal problem
sigma = 10.0 # eigenvalue guess, use closed-form solution
nEigs = 3
MP = fea_solver.createModalProblem("platform-modal-problem", sigma, nEigs)

MP.solve()
# MP.evalFunctions(tacs_funcs, evalFuncs=function_names)
# MP.evalFunctionsSens(tacs_sens, evalFuncs=function_names)
MP.writeSolution(
    baseName="tacs_output", outputDir=tacs_model.analysis_dir(proc=0)
)
