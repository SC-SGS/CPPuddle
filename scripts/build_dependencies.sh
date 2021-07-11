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

${SCRIPTS_DIR}/build_hpx.sh
${SCRIPTS_DIR}/build_kokkos.sh
${SCRIPTS_DIR}/build_hpx_kokkos.sh
