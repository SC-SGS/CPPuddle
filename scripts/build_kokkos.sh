#!/usr/bin/env bash

if [ -z "${APPEND_DIRNAME}" ]; then
  echo "build_kokkos.sh is meant to be called via build_dependencies.sh which sets the correct variables and loads the machine configs"
  echo "Exiting..."
  exit 1
fi

set -euxo

SOURCE_DIR=${SCRIPTS_DIR}/../external_dependencies/kokkos
BUILD_DIR=${SCRIPTS_DIR}/../external_dependencies/build/kokkos-${APPEND_DIRNAME}
INSTALL_DIR=${SCRIPTS_DIR}/../external_dependencies/install/kokkos-${APPEND_DIRNAME}

mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}
if [[ "${CXX}" == "clang++" ]]; then # clang too old on our usual machine - compile without CUDA
  cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DHPX_DIR=${HPX_ROOT} -DKokkos_CXX_STANDARD=14 ${CURRENT_CUDA_ARCH_FLAG} -DKokkos_ARCH_HSW=ON -DKokkos_ENABLE_HPX=ON -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_TESTS=OFF -DKokkos_ENABLE_HPX_ASYNC_DISPATCH=ON -DKokkos_ENABLE_INTERNAL_FENCES=OFF -DKokkos_ENABLE_CUDA_LAMBDA=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ${SOURCE_DIR}
else
  #cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DHPX_DIR=${HPX_ROOT} -DKokkos_CXX_STANDARD=14 ${CURRENT_CUDA_ARCH_FLAG} -DKokkos_ARCH_HSW=ON -DKokkos_ENABLE_HPX=ON -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_TESTS=OFF -DKokkos_ENABLE_HPX_ASYNC_DISPATCH=ON -DKokkos_ENABLE_INTERNAL_FENCES=OFF -DKokkos_ENABLE_CUDA_LAMBDA=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ${SOURCE_DIR}
  cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DHPX_DIR=${HPX_ROOT} -DKokkos_CXX_STANDARD=14 ${CURRENT_CUDA_ARCH_FLAG} -DKokkos_ARCH_HSW=ON -DKokkos_ENABLE_HPX=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_TESTS=OFF -DKokkos_ENABLE_HPX_ASYNC_DISPATCH=ON -DKokkos_ENABLE_INTERNAL_FENCES=OFF -DKokkos_ENABLE_CUDA_LAMBDA=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ${SOURCE_DIR}
fi
make -j$(nproc) install
popd
