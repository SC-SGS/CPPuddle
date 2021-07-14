#!/usr/bin/env bash

if [ -z "${APPEND_DIRNAME}" ]; then
  echo "build_hpx_kokkos.sh is meant to be called via build_dependencies.sh which sets the correct variables and loads the machine configs"
  echo "Exiting..."
  exit 1
fi

set -euxo

SOURCE_DIR=${SCRIPTS_DIR}/../external_dependencies/hpx-kokkos
BUILD_DIR=${SCRIPTS_DIR}/../external_dependencies/build/hpx-kokkos-${APPEND_DIRNAME}
INSTALL_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-kokkos-${APPEND_DIRNAME}

mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}
cmake -DHPX_KOKKOS_ENABLE_TESTS=OFF -DHPX_KOKKOS_ENABLE_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ${SOURCE_DIR}
make -j$(nproc) install
popd

