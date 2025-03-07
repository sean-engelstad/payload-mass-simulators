from tacs import caps2tacs
from mpi4py import MPI
from funtofem import *

# run a steady elastic structural analysis in TACS using the tacsAIM wrapper caps2tacs submodule
# -------------------------------------------------------------------------------------------------
# 1: build the tacs aim, egads aim wrapper classes

comm = MPI.COMM_WORLD
f2f_model = FUNtoFEMmodel("beam")
wing = Body.aeroelastic(
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
for capsGroup in ["base", "endmass"]: #"root", "tip"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=0.01
    ).register_to(tacs_model)
    Variable.structural(capsGroup, value=0.01).set_bounds(
        lower=1e-4, upper=0.1, scale=100
    )
for iskin in range(1, nskin+1):
    capsGroup = f"skin{iskin}"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=0.01
    ).register_to(tacs_model)
    Variable.structural(capsGroup, value=0.01).set_bounds(
        lower=1e-4, upper=0.1, scale=100
    )
for istiff in range(1, nstiff+1):
    capsGroup = f"stiff{istiff}"
    caps2tacs.ShellProperty(
        caps_group=capsGroup, material=aluminum, membrane_thickness=0.01
    ).register_to(tacs_model)
    Variable.structural(capsGroup, value=0.01).set_bounds(
        lower=1e-4, upper=0.1, scale=100
    )
# later we can add in extra thickness variables, etc.
# such as each side of the skin (all 4 sides)

caps2tacs.PinConstraint("base").register_to(tacs_model)
caps2tacs.GridForce("loading", direction=[0, 0, 1.0], magnitude=10).register_to(tacs_model)

f2f_model.structural = tacs_model

# SHAPE VARIABLES
for direc in ["x", "z"]:
    Variable.shape(f"beam:root_d{direc}", value=1.0).set_bounds(
        lower=0.1, upper=10.0
    ).register_to(wing)
    Variable.shape(f"beam:taper_{direc}", value=0.5).set_bounds(
        lower=0.01, upper=1.5
    ).register_to(wing)
Variable.shape(f"beam:length", value=2.0).set_bounds(
    lower=0.1, upper=10.0
).register_to(wing)

for direc in ["dx", "dy", "dz"]:
    Variable.shape(f"endmass:{direc}", value=0.6).set_bounds(
        lower=0.1, upper=3.0
    ).register_to(wing)

for mystr in ["poly", "hole_poly"]:
    Variable.shape(f"stiffener:{mystr}B", value=1.0).set_bounds(
        lower=0.6, upper=1.4
    ).register_to(wing)
    Variable.shape(f"stiffener:{mystr}C", value=0.0).set_bounds(
        lower=-0.3, upper=0.3
    ).register_to(wing)

# register the funtofem Body to the model
wing.register_to(f2f_model)

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
MP = fea_solver.createModalProblem("beam-modal-problem", sigma, nEigs)

MP.solve()
# MP.evalFunctions(tacs_funcs, evalFuncs=function_names)
# MP.evalFunctionsSens(tacs_sens, evalFuncs=function_names)
MP.writeSolution(
    baseName="tacs_output", outputDir=tacs_model.analysis_dir(proc=0)
)

