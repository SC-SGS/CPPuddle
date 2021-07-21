#!/usr/bin/env bash

set -euxo
#/
#/ -------------------------------------------------------
#/ Configure CPPuddle Build Directory
#/ -------------------------------------------------------
#/ Usage: build_dependencies.sh build_type compiler
#/ Arguments:
#/ -> build_type: [Release|RelWithDebInfo|Debug]
#/ -> compiler: [gcc|clang]
#/
usage() {
    grep '^#/' "$0" | cut -c4-
    exit 0
}
expr "$*" : ".*--help" > /dev/null && usage

if [[ $# != 2 ]]; then
  usage
fi


# Stolen from https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself#59916
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Load machine configs
source ${SCRIPTS_DIR}/machine_configs.sh 
case $(hostname) in
    pcsgs0*)
        source_config_pcsgs0x ${SCRIPTS_DIR} "$1" "$2"
        ;;
    daint)
        source_config_daint ${SCRIPTS_DIR} "$1" "$2"
        ;;
    *)
        source_config_default ${SCRIPTS_DIR} "$1" "$2"
        ;;
esac

BUILD_DIR=${SCRIPTS_DIR}/../build/${APPEND_DIRNAME}
INSTALL_DIR=${SCRIPTS_DIR}/../install/${APPEND_DIRNAME}
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}
pushd ${BUILD_DIR}
# TODO Install newer clang on pcsgs04
if [[ "${CXX}" == "clang++" ]]; then # clang too old on our usual machine - compile without CUDA
  cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON -DCPPUDDLE_WITH_TESTS=ON -DCPPUDDLE_WITH_HPX=ON -DCPPUDDLE_WITH_CUDA=OFF -DCPPUDDLE_WITH_KOKKOS=OFF -DCPPUDDLE_WITH_COUNTERS=ON -DHPX_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${APPEND_DIRNAME}/lib/cmake/HPX -DKokkos_DIR=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${APPEND_DIRNAME}/lib/cmake/Kokkos -DHPXKokkos_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-kokkos-${APPEND_DIRNAME}/lib/cmake/HPXKokkos ../..
else
  cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON -DCPPUDDLE_WITH_TESTS=ON -DCPPUDDLE_WITH_HPX=ON -DCPPUDDLE_WITH_CUDA=ON -DCPPUDDLE_WITH_KOKKOS=ON -DCPPUDDLE_WITH_COUNTERS=ON -DHPX_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-${APPEND_DIRNAME}/lib/cmake/HPX -DKokkos_DIR=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${APPEND_DIRNAME}/lib/cmake/Kokkos -DHPXKokkos_DIR=${SCRIPTS_DIR}/../external_dependencies/install/hpx-kokkos-${APPEND_DIRNAME}/lib/cmake/HPXKokkos ../..
fi
popd

