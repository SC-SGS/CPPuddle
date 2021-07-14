#!/usr/bin/env bash

if [ -z "${APPEND_DIRNAME}" ]; then
  echo "build_hpx.sh is meant to be called via build_dependencies.sh which sets the correct variables and loads the machine configs"
  echo "Exiting..."
  exit 1
fi

set -euxo

SOURCE_DIR=${SCRIPTS_DIR}/../external_dependencies/hpx
BUILD_DIR=${SCRIPTS_DIR}/../external_dependencies/build/hpx-${APPEND_DIRNAME}
INSTALL_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${APPEND_DIRNAME}

HPX_MALLOC=system

# NOTE: We do not use nvcc_wrapper to build all of HPX.
mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}
cmake -DCMAKE_CXX_COMPILER=${HPX_COMPILER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_TESTS=OFF -DHPX_WITH_CXX14=ON -DHPX_WITH_CUDA=ON -DHPX_WITH_MALLOC=${HPX_MALLOC} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ${SOURCE_DIR} 
make VERBOSE=1 -j$(nproc) install
popd
