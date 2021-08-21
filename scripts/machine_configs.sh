#!/usr/bin/env bash


function source_config_pcsgs0x() {
  SCRIPTS_DIR="$1"
  export CMAKE_BUILD_TYPE="$2"
  if [[ "${3}" == "gcc" ]]; then
    export CXX=${SCRIPTS_DIR}/../external_dependencies/kokkos/bin/nvcc_wrapper
    export HPX_COMPILER="g++"
  elif [[ "${3}" == "clang" ]]; then
    export CXX=clang++
    export HPX_COMPILER=clang++
  else
    echo "Invalid compiler!"
    exit 1
  fi
  export APPEND_DIRNAME="$3-$2"


  #export CXX=${SCRIPTS_DIR}/../external_dependencies/kokkos/bin/nvcc_wrapper
  export CXXFLAGS="-Wno-cpp" # Silence deprecated header warnings in HPX
  #export CMAKE_BUILD_TYPE=Release
  export HPX_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${APPEND_DIRNAME}/lib/cmake/HPX
  export Kokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${APPEND_DIRNAME}/lib/cmake/Kokkos
  export HPXKokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-hpx-interop-${APPEND_DIRNAME}/lib/cmake/HPXKokkos

  export CURRENT_CUDA_ARCH_FLAG="-DKokkos_ARCH_SKX=ON -DKokkos_ARCH_AMPERE80=ON"
}

function source_config_daint() {
  echo -e "\033[33mWARNING: Untested configuration...\033[0m"
  echo -e "\033[33mWARNING: Remove exit 1 in machine_configs.sh if you really want to try this\033[0m"
  exit 1
  # TODO Test this on daint

  SCRIPTS_DIR="$1"
  export CMAKE_BUILD_TYPE="$2"
  if [[ "${3}" == "gcc" ]]; then
    export CXX=${SCRIPTS_DIR}/../external_dependencies/kokkos/bin/nvcc_wrapper
    export HPX_COMPILER="g++"
  elif [[ "${3}" == "clang" ]]; then
    export CXX=clang++
    export HPX_COMPILER=clang++
  else
    echo "Invalid compiler!"
    exit 1
  fi
  export APPEND_DIRNAME="$3-$2"

  export CRAYPE_LINK_TYPE=dynamic
  export CXXFLAGS="-Wno-cpp" # Silence deprecated header warnings in HPX
  export HPX_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${APPEND_DIRNAME}/lib64/cmake/HPX
  export Kokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${APPEND_DIRNAME}/lib64/cmake/Kokkos
  export HPXKokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-hpx-interop-${APPEND_DIRNAME}/lib64/cmake/HPXKokkos

  CURRENT_CUDA_ARCH_FLAG="-D Kokkos_ARCH_HSW=ON -DKokkos_ARCH_PASCAL60=ON"

  module load daint-gpu
  module load CMake
  module switch PrgEnv-cray PrgEnv-gnu
  module load cudatoolkit
  module load Boost
  module load cray-jemalloc
}

function source_config_xp14() {
  SCRIPTS_DIR="$1"
  export CMAKE_BUILD_TYPE="$2"
  if [[ "${3}" == "gcc" ]]; then
    export CXX=${SCRIPTS_DIR}/../external_dependencies/kokkos/bin/nvcc_wrapper
    export HPX_COMPILER="g++"
  elif [[ "${3}" == "clang" ]]; then
    export CXX=clang++
    export HPX_COMPILER=clang++
  else
    echo "Invalid compiler!"
    exit 1
  fi
  export APPEND_DIRNAME="-$3-$2"


  #export CXX=${SCRIPTS_DIR}/../external_dependencies/kokkos/bin/nvcc_wrapper
  export CXXFLAGS="-Wno-cpp" # Silence deprecated header warnings in HPX
  #export CMAKE_BUILD_TYPE=Release
  export HPX_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${APPEND_DIRNAME}/lib/cmake/HPX
  export Kokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${APPEND_DIRNAME}/lib/cmake/Kokkos
  export HPXKokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-hpx-interop-${APPEND_DIRNAME}/lib/cmake/HPXKokkos

  export CURRENT_CUDA_ARCH_FLAG="-DKokkos_ARCH_SKX=ON -DKokkos_ARCH_TURING75=ON"
}

function source_config_default() {
  echo -e "\033[33mWARNING: Default configuration... You likely need to modify this\033[0m"
  sleep 8 

  SCRIPTS_DIR="$1"
  export CMAKE_BUILD_TYPE="$2"
  if [[ "${3}" == "gcc" ]]; then
    export CXX=${SCRIPTS_DIR}/../external_dependencies/kokkos/bin/nvcc_wrapper
    export HPX_COMPILER="g++"
  elif [[ "${3}" == "clang" ]]; then
    export CXX=clang++
    export HPX_COMPILER=clang++
  else
    echo "Invalid compiler!"
    exit 1
  fi
  export APPEND_DIRNAME="-$3-$2"


  #export CXX=${SCRIPTS_DIR}/../external_dependencies/kokkos/bin/nvcc_wrapper
  export CXXFLAGS="-Wno-cpp" # Silence deprecated header warnings in HPX
  #export CMAKE_BUILD_TYPE=Release
  export HPX_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${APPEND_DIRNAME}/lib/cmake/HPX
  export Kokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${APPEND_DIRNAME}/lib/cmake/Kokkos
  export HPXKokkos_ROOT=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-hpx-interop-${APPEND_DIRNAME}/lib/cmake/HPXKokkos

  export CURRENT_CUDA_ARCH_FLAG="-DKokkos_ARCH_HSW=ON -DKokkos_ARCH_PASCAL61=ON"
}
