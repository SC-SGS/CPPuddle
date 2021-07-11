#!/usr/bin/env bash

set -euxo

# Stolen from https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself#59916
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

case $(hostname) in
    pcsgs*)
        source ${SCRIPTS_DIR}/spack_env.sh
        ;;
    *)
        source ${SCRIPTS_DIR}/daint_env.sh
        ;;
esac

BUILD_DIR=${SCRIPTS_DIR}/../build/${CMAKE_BUILD_TYPE}
INSTALL_DIR=${SCRIPTS_DIR}/../install/${CMAKE_BUILD_TYPE}
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}
pushd ${BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON -DCPPUDDLE_WITH_TESTS=ON -DCPPUDDLE_WITH_HPX=ON -DCPPUDDLE_WITH_CUDA=ON -DCPPUDDLE_WITH_KOKKOS=ON -DCPPUDDLE_WITH_COUNTERS=ON -DHPX_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${CMAKE_BUILD_TYPE}/lib/cmake/HPX -DKokkos_DIR=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${CMAKE_BUILD_TYPE}/lib/cmake/Kokkos -DHPXKokkos_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-kokkos-${CMAKE_BUILD_TYPE}/lib/cmake/HPXKokkos ../..
popd

