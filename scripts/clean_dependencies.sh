#!/usr/bin/env bash

set -euxo

# Stolen from https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself#59916
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source ${SCRIPTS_DIR}/daint_env.sh

rm -rf ${SCRIPTS_DIR}/../external_dependencies/hpx-install-${CMAKE_BUILD_TYPE}
rm -rf ${SCRIPTS_DIR}/../external_dependencies/kokkos-install-${CMAKE_BUILD_TYPE}

