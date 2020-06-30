export CORE_INSTALL_DIR="/lore/mohara/core_INSTALL_sim15/"
export MFEM_LIB_DIR="/lore/mohara/mfemFT_INSTALL/lib"
export MFEM_INCLUDE_DIR="/lore/mohara/mfemFT_INSTALL/include"
export CORE_INCLUDE_DIR="/lore/mohara/core_INSTALL_sim15/include"

flags="-g -O0 -std=c++11"
cmake .. \
  -DCMAKE_C_COMPILER="mpicc" \
  -DCMAKE_CXX_COMPILER="mpicxx" \
  -DCMAKE_C_FLAGS="${flags}" \
  -DCMAKE_CXX_FLAGS="${flags}" \
  -DCMAKE_EXE_LINKER_FLAGS="-lpthread ${flags}" \
  -DENABLE_SIMMETRIX=OFF \
  -DSIM_PARASOLID=ON \
  -DSCOREC_PREFIX="${CORE_INSTALL_DIR}" \
  -DMFEM_LIB_DIR="${MFEM_LIB_DIR}" \
  -DMFEM_INCLUDE_DIR="${MFEM_INCLUDE_DIR}" \
  -DPUMI_INCLUDE_DIR="${CORE_INCLUDE_DIR}"
