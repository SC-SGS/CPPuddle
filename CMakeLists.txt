cmake_minimum_required(VERSION 3.13)

project(hpx_kokkos_example CXX CUDA)

# TODO: We actually require unreleased features. The minimum versions need to
# be updated when they are released.
find_package(HPX 1.4.1 REQUIRED)
find_package(Kokkos 3.0.0 REQUIRED)

# Check that Kokkos and HPX options are consistent.
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
  if(NOT HPX_WITH_CUDA)
    message(FATAL_ERROR "Kokkos was built with CUDA support, HPX was not")
  endif()
else()
  if(HPX_WITH_CUDA)
    message(FATAL_ERROR "HPX was built with CUDA support, Kokkos was not")
  endif()
endif()

kokkos_check(DEVICES HPX)
kokkos_check(OPTIONS HPX_ASYNC_DISPATCH)

## Interface targets
# TODO: kokkos-hpx-interop should be installable.
if(NOT KOKKOS_HPX_INTEROP_ROOT)
  message(FATAL_ERROR "KOKKOS_HPX_INTEROP_ROOT is not set")
endif()
add_library(kokkos_hpx_interop INTERFACE)
target_include_directories(kokkos_hpx_interop INTERFACE ${KOKKOS_HPX_INTEROP_ROOT}/include)

add_library(buffer_manager INTERFACE)
target_include_directories(buffer_manager INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

## Tests
add_executable(allocator_test src/allocator_test.cpp)
target_link_libraries(allocator_test
  PRIVATE HPX::hpx Kokkos::kokkos kokkos_hpx_interop buffer_manager)

add_executable(
  allocator_kokkos_test
  src/allocator_kokkos_test.cpp
  include/buffer_manager.hpp)
target_link_libraries(allocator_kokkos_test
  PRIVATE HPX::hpx Kokkos::kokkos kokkos_hpx_interop buffer_manager)

add_executable(interop ${KOKKOS_HPX_INTEROP_ROOT}/src/kokkos_hpx.cpp)
target_link_libraries(interop PRIVATE
  PRIVATE HPX::hpx Kokkos::kokkos kokkos_hpx_interop buffer_manager)

add_executable(allocator_cuda_test src/allocator_cuda_test.cu)
target_link_libraries(allocator_cuda_test
  PRIVATE HPX::hpx Kokkos::kokkos kokkos_hpx_interop buffer_manager)

add_executable(
  allocator_kokkos_executor_stream_test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/allocator_kokkos_executor_stream_test.cpp)
target_link_libraries(allocator_kokkos_executor_stream_test
  PRIVATE HPX::hpx Kokkos::kokkos kokkos_hpx_interop buffer_manager)

add_executable(
  allocator_executor_stream_test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/allocator_executor_stream_test.cpp)
target_link_libraries(allocator_executor_stream_test
  PRIVATE HPX::hpx Kokkos::kokkos kokkos_hpx_interop buffer_manager)
