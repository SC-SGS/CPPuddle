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
    ????book)
        source_config_xp14 ${SCRIPTS_DIR} "$1" "$2"
        ;;
    toranj*)
        source_config_toranj ${SCRIPTS_DIR} "$1" "$2"
        ;;
    buran*)
        source_config_buran ${SCRIPTS_DIR} "$1" "$2"
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
# TODO Reactivate CUDA/KOKKOS once we have a newer cmake version on the test machine
if [[ "${CXX}" == "clang++" ]]; then # clang/cmake too old on our usual machine - compile without CUDA
  cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCPPUDDLE_WITH_TESTS=ON -DCPPUDDLE_WITH_HPX=OFF -DCPPUDDLE_WITH_CUDA=OFF -DCPPUDDLE_WITH_KOKKOS=OFF -DCPPUDDLE_WITH_COUNTERS=ON -DHPX_DIR=${HPX_ROOT} -DKokkos_DIR=${Kokkos_ROOT} -DHPXKokkos_DIR=${HPXKokkos_ROOT} ../..
else
  cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCPPUDDLE_WITH_TESTS=ON -DCPPUDDLE_WITH_HPX=OFF -DCPPUDDLE_WITH_CUDA=OFF -DCPPUDDLE_WITH_KOKKOS=OFF -DCPPUDDLE_WITH_COUNTERS=ON -DHPX_DIR=${HPX_ROOT} -DKokkos_DIR=${Kokkos_ROOT} -DHPXKokkos_DIR=${HPXKokkos_ROOT} ../..
fi
popd
cp ${BUILD_DIR}/compile_commands.json ${SCRIPTS_DIR}/../compile_commands.json

