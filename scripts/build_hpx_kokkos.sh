#!/usr/bin/env bash

set -euxo

# Stolen from https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself#59916
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source ${SCRIPTS_DIR}/daint_env.sh

SOURCE_DIR=${SCRIPTS_DIR}/../external_dependencies/kokkos-hpx-interop
BUILD_DIR=${SCRIPTS_DIR}/../external_dependencies/build/kokkos-hpx-interop-${CMAKE_BUILD_TYPE}
INSTALL_DIR=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-hpx-interop-${CMAKE_BUILD_TYPE}

mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ${SOURCE_DIR}
make -j$(nproc) install
popd

