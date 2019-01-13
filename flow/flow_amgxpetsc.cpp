// only compile blackoil option
#define FLOW_SINGLE_PURPOSE 1
#define ENABLE_FLOW_BLACKOIL 1
// choose solvers
#define USE_AMGX_SOLVERS 1
#define FLOW_USE_DUNE_FEM_PETSC 1
#include "flow.cpp"
