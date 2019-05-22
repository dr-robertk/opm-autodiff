// only compile blackoil option
#define FLOW_SINGLE_PURPOSE 1
#define ENABLE_FLOW_BLACKOIL 1
// petsc solvers
#define FLOW_USE_DUNE_FEM_PETSC 1

#include "flow.cpp"
#include "flow_ebos_blackoil.cpp"
