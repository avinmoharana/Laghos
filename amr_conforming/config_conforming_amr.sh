export CORE_INSTALL_DIR=path_to_core_install_dir
export MFEM_LIB_DIR=path_to_fem_lib_dir
export MFEM_INCLUDE_DIR=path_to_mfem_include_dir
export CORE_INCLUDE_DIR=path_to_core_include_dir

flags="-g -O0 -std=c++11"
cmake .. \
  -DCMAKE_C_COMPILER="mpicc" \
  -DCMAKE_CXX_COMPILER="mpicxx" \
  -DCMAKE_C_FLAGS="${flags}" \
  -DCMAKE_CXX_FLAGS="${flags}" \
  -DCMAKE_EXE_LINKER_FLAGS="-lpthread ${flags}" \
  -DENABLE_SIMMETRIX=OFF \
  -DSIM_PARASOLID=OFF \
  -DSCOREC_PREFIX="${CORE_INSTALL_DIR}" \
  -DMFEM_LIB_DIR="${MFEM_LIB_DIR}" \
  -DMFEM_INCLUDE_DIR="${MFEM_INCLUDE_DIR}" \
  -DPUMI_INCLUDE_DIR="${CORE_INCLUDE_DIR}"
