#!/usr/bin/env bash

set -euxo

# Stolen from https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself#59916
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source ${SCRIPTS_DIR}/daint_env.sh

SOURCE_DIR=${SCRIPTS_DIR}/../external_dependencies/hpx
BUILD_DIR=${SCRIPTS_DIR}/../external_dependencies/build/hpx-${CMAKE_BUILD_TYPE}
INSTALL_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${CMAKE_BUILD_TYPE}

HPX_MALLOC=system
if [[ "${CMAKE_BUILD_TYPE}" == "Release" ]]; then
    HPX_MALLOC=jemalloc
fi

# NOTE: We do not use nvcc_wrapper to build all of HPX.
mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}
cmake -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_CXX14=ON -DHPX_WITH_CUDA=ON -DHPX_WITH_MALLOC=${HPX_MALLOC} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ${SOURCE_DIR}
make -j$(nproc) install
popd
