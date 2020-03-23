#!/usr/bin/env bash

set -euxo

# Stolen from https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself#59916
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source ${SCRIPTS_DIR}/daint_env.sh

BUILD_DIR=${SCRIPTS_DIR}/../build/${CMAKE_BUILD_TYPE}
mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON ../..
popd

