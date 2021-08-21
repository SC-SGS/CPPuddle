#!/usr/bin/env bash

set -euxo
#/
#/ -------------------------------------------------------
#/ Dependency builder:
#/ -------------------------------------------------------
#/ Builds hpx kokkos hpx-kokkos dependencies of CPPuddle.
#/ Machine configs are loaded from machine_configs.sh
#/ If your machine is not in there you likely need to adapt
#/ the scripts to make the build work
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
export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Load machine configs
source ${SCRIPTS_DIR}/machine_configs.sh 
case $(hostname) in
    pcsgs0*)
        source_config_pcsgs0x ${SCRIPTS_DIR} "$1" "$2"
        ;;
    daint)
        source_config_daint ${SCRIPTS_DIR} "$1" "$2"
        ;;
    xp14x2)
        source_config_xp14 ${SCRIPTS_DIR} "$1" "$2"
        ;;
    *)
        source_config_default ${SCRIPTS_DIR} "$1" "$2"
        ;;
esac

${SCRIPTS_DIR}/build_hpx.sh
${SCRIPTS_DIR}/build_kokkos.sh
${SCRIPTS_DIR}/build_hpx_kokkos.sh
