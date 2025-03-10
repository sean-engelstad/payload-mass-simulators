## [import]
# Import pyCAPS module
import pyCAPS

# Import os module
import os
import shutil
import argparse
## [import]

# Create project name
projectName = "TacsBeam"

# Working directory
workDir = projectName
# workDir = os.path.join(str(args.workDir[0]), projectName)

## [initateProblem]
# Initialize CAPS Problem
capsProblem = pyCAPS.Problem(problemName=workDir,
                             capsFile='simple-beam.csm',
                             outLevel=1)
## [geometry]


# Load egadsTess aim
egads = capsProblem.analysis.create(aim = "egadsTessAIM")

# Set meshing parameters
egads.input.Edge_Point_Max = 2
egads.input.Edge_Point_Min = 2

egads.input.Tess_Params = [.05,.5,15]


## [loadAIM]
# Load tacs aim
tacs = capsProblem.analysis.create(aim = "tacsAIM",
                                      name = "tacs")
## [loadAIM]

## [setInputs]
tacs.input["Mesh"].link(egads.output["Surface_Mesh"])
tacs.input.Proj_Name = "tacs_test"
tacs.input.File_Format = "Free"
tacs.input.Mesh_File_Format = "Large"
tacs.input.Analysis_Type = "Static"
## [setInputs]

## [defineMaterials]
madeupium    = {"materialType" : "isotropic",
                "youngModulus" : 1.0E7 ,
                "poissonRatio" : .33,
                "density"      : 0.1}

tacs.input.Material = {"Madeupium": madeupium}
## [defineMaterials]

## [defineProperties]
rod  =   {"propertyType"      : "Bar",
          "material"          : "Madeupium",
          "crossSecArea"      : 1.0,
          "zAxisInertia" : 1.0e4,
          "yAxisInertia" : 1.0e4,}

capsGroup1 = "beam"
tacs.input.Property = {capsGroup1: rod,}
## [defineProperties]

## [defineConstraints]
# Set constraints
capsConstraint = "root"
conOne = {"groupName"         : [capsConstraint],
          "dofConstraint"     : 123456}

tacs.input.Constraint = {"root-constr": conOne}
## [defineConstraints]

## [defineLoad]
capsLoad = "tip"
loadOne = {"groupName"         : capsLoad,
        "loadType"          : "GridForce",
        "forceScaleFactor"  : 20000.0,
        "directionVector"   : [0.8, 0.0, 0.6]}

tacs.input.Load = {"loadOne": loadOne,}

## [defineLoad]

## [defineAnalysis]
caseOne = {"analysisType"         : "Static",
           "analysisConstraint"     : "conOne",
           "analysisLoad"         : "loadOne"}

tacs.input.Analysis = {"caseOne": caseOne,}
## [defineAnalysis]

# Run AIM pre-analysis
## [preAnalysis]
tacs.preAnalysis()
## [preAnalysis]

## [run]
print ("\n\nRunning tacs......")
# TBD add running stuff here

print ("Done running tacs!")
## [run]

## [postAnalysis]
# tacs.postAnalysis()
## [postAnalysis]
