#!/usr/bin/env bash

# Stolen from https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself#59916
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CXX=${SCRIPTS_DIR}/../external_dependencies/kokkos/bin/nvcc_wrapper
export CXXFLAGS="-Wno-cpp" # Silence deprecated header warnings in HPX
export CMAKE_BUILD_TYPE=Release
export HPX_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${CMAKE_BUILD_TYPE}/lib/cmake/HPX
export Kokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${CMAKE_BUILD_TYPE}/lib/cmake/Kokkos
export HPXKokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-hpx-interop-${CMAKE_BUILD_TYPE}/lib/cmake/HPXKokkos
CURRENT_CUDA_ARCH_FLAG="-DKokkos_ARCH_PASCAL61=ON"
