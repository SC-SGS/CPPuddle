cmake_minimum_required(VERSION 3.13)

project(hpx_kokkos_example CXX) 

# TODO: We actually require unreleased features. The minimum versions need to
# be updated when they are released.
find_package(HPX 1.4.1 REQUIRED)
find_package(Kokkos 3.0.0 REQUIRED)
find_package(HPXKokkos REQUIRED) # No version yet.

option(INTEROPT_WITH_CUDA "Enable CUDA" ON)

if (INTEROPT_WITH_CUDA)
   enable_language(CUDA) 
endif ()

# Check that Kokkos and HPX options are consistent.
if(Kokkos_ENABLE_CUDA)
  if(NOT HPX_WITH_CUDA)
    message(FATAL_ERROR "Kokkos was built with CUDA support, HPX was not")
  endif()
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

# Linter
find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-header-filter=.;-checks=*,-fuchsia*,-misc-unused-parameters,-readability-named-parameter,-cert*,-android*,-modernize-pass-by-value")
endif()

# Format
file(GLOB_RECURSE
     ALL_CXX_SOURCE_FILES
     src/*.cpp include/*.hpp 
     )
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(
    clang-format
    COMMAND clang-format
    -i
    #-style=llvm
    -style=file # checks .clang-format file
    ${ALL_CXX_SOURCE_FILES}
    )
endif()


## Interface targets
add_library(buffer_manager INTERFACE)
target_include_directories(buffer_manager INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


## Tests
add_executable(allocator_test src/allocator_test.cpp)
target_link_libraries(allocator_test
  PRIVATE HPX::hpx buffer_manager)

add_executable(allocator_hpx_test src/allocator_hpx_test.cpp)
target_link_libraries(allocator_hpx_test
  PRIVATE HPX::hpx buffer_manager)

if (INTEROPT_WITH_CUDA)
  add_executable(
    allocator_kokkos_test
    src/allocator_kokkos_test.cpp
    include/buffer_manager.hpp)
  target_link_libraries(allocator_kokkos_test
    PRIVATE HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
  target_compile_definitions(allocator_kokkos_test PRIVATE INTEROPT_HAVE_CUDA)

  add_executable(allocator_cuda_test src/allocator_cuda_test.cu)
  target_link_libraries(allocator_cuda_test
    PRIVATE HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
  target_compile_definitions(allocator_cuda_test PRIVATE INTEROPT_HAVE_CUDA)

  add_executable(
    allocator_kokkos_executor_stream_test
    ${CMAKE_CURRENT_SOURCE_DIR}/src/allocator_kokkos_executor_stream_test.cpp)
  target_link_libraries(allocator_kokkos_executor_stream_test
    PRIVATE HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
  target_compile_definitions(allocator_kokkos_executor_stream_test PRIVATE INTEROPT_HAVE_CUDA)

  add_executable(
    allocator_executor_stream_test
    ${CMAKE_CURRENT_SOURCE_DIR}/src/allocator_executor_stream_test.cpp)
  target_link_libraries(allocator_executor_stream_test
    PRIVATE HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
  target_compile_definitions(allocator_executor_stream_test PRIVATE INTEROPT_HAVE_CUDA)

  add_executable(
    minimal_parallel_for
    ${CMAKE_CURRENT_SOURCE_DIR}/src/minimal_parallel_for.cpp)
  target_link_libraries(minimal_parallel_for
    PRIVATE HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
  target_compile_definitions(minimal_parallel_for PRIVATE INTEROPT_HAVE_CUDA)
endif()

# install the headers only
install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
