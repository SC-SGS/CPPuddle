cmake_minimum_required(VERSION 3.13)

project(CPPuddle CXX) # Some random project name

# TODO: We actually require unreleased features. The minimum versions need to
# be updated when they are released.
find_package(HPX 1.4.1 REQUIRED)
find_package(Kokkos 3.0.0 REQUIRED)
find_package(HPXKokkos REQUIRED) # No version yet.

option(INTEROPT_WITH_CUDA "Enable CUDA" ON)
option(INTEROPT_WITH_TESTS "Build tests" OFF)

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
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-header-filter=.;-checks=*,-fuchsia*,-misc-unused-parameters,-readability-named-parameter,-cert*,-android*,-modernize-pass-by-value,-hicpp-use-equals-default,-modernize-use-equals-default")
endif()

# Format
file(GLOB_RECURSE
     ALL_CXX_SOURCE_FILES
     src/*.cpp include/*.hpp src/*.cu
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
add_library(buffer_manager SHARED src/buffer_manager_definitions.cpp)
target_include_directories(buffer_manager INTERFACE
 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
 )

add_library(stream_manager SHARED src/stream_manager_definitions.cpp)
target_link_libraries(stream_manager
    PRIVATE HPX::hpx)
target_include_directories(stream_manager INTERFACE 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> 
)

# install libs with the defitions:
install(TARGETS buffer_manager EXPORT CPPuddle
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
)
install(TARGETS stream_manager EXPORT CPPuddle
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  
)
# install all headers
install(
  DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES cppuddle-config.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CPPuddle/)
install(EXPORT CPPuddle DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CPPuddle/)




## Tests
if (INTEROPT_WITH_TESTS)
add_executable(allocator_test src/allocator_test.cpp)
target_link_libraries(allocator_test
  HPX::hpx buffer_manager)

add_executable(allocator_aligned_test src/allocator_aligned_test.cpp)
target_link_libraries(allocator_aligned_test
  HPX::hpx buffer_manager)

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
    PRIVATE Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
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

  add_executable(
    stream_test
    src/stream_test.cpp
    include/stream_manager.hpp)
  target_link_libraries(stream_test
    PRIVATE HPX::hpx buffer_manager stream_manager)
  target_compile_definitions(stream_test PRIVATE INTEROPT_HAVE_CUDA)
endif()


enable_testing()

# Basic functionality tests
add_test(allocator_test.run allocator_test --arraysize 5000000 --passes 200 --outputfile allocator_test.out)
set_tests_properties(allocator_test.run PROPERTIES
  FIXTURES_SETUP allocator_test_output
)
add_test(allocator_test.analyse_recycle_rate cat allocator_test.out)
set_tests_properties(allocator_test.analyse_recycle_rate PROPERTIES
  FIXTURES_REQUIRED allocator_test_output
  PASS_REGULAR_EXPRESSION "==> Recycle rate: [ ]* 99.5%"
)
add_test(allocator_test.analyse_marked_buffers_cleanup cat allocator_test.out)
set_tests_properties(allocator_test.analyse_marked_buffers_cleanup PROPERTIES
  FIXTURES_REQUIRED allocator_test_output
  PASS_REGULAR_EXPRESSION "--> Number of buffers that were marked as used upon cleanup:[ ]* 0"
)
add_test(allocator_test.analyse_cleaned_buffers cat allocator_test.out)
set_tests_properties(allocator_test.analyse_cleaned_buffers PROPERTIES
  FIXTURES_REQUIRED allocator_test_output
  PASS_REGULAR_EXPRESSION "--> Number cleaned up buffers:[ ]* 1"
)
add_test(allocator_test.analyse_created_buffers cat allocator_test.out)
set_tests_properties(allocator_test.analyse_created_buffers PROPERTIES
  FIXTURES_REQUIRED allocator_test_output
  PASS_REGULAR_EXPRESSION "--> Number of times a new buffer had to be created for a request:[ ]* 1"
)
add_test(allocator_test.analyse_bad_allocs cat allocator_test.out)
set_tests_properties(allocator_test.analyse_bad_allocs PROPERTIES
  FIXTURES_REQUIRED allocator_test_output
  PASS_REGULAR_EXPRESSION "--> Number of bad_allocs that triggered garbage collection: [ ]* 0"
)
if (NOT CMAKE_BUILD_TYPE MATCHES "Debug") # Performance tests only make sense with optimizations on
  add_test(allocator_test.performance.analyse_recycle_performance cat allocator_test.out)
  set_tests_properties(allocator_test.performance.analyse_recycle_performance PROPERTIES
    FIXTURES_REQUIRED allocator_test_output
    PASS_REGULAR_EXPRESSION "Test information: Recycler was faster than default allocator!"
  )
  add_test(allocator_test.performance.analyse_aggressive_performance cat allocator_test.out)
  set_tests_properties(allocator_test.performance.analyse_aggressive_performance PROPERTIES
    FIXTURES_REQUIRED allocator_test_output
    PASS_REGULAR_EXPRESSION "Test information: Recycler was faster than default allocator!"
  )
endif()
add_test(allocator_test.fixture_cleanup ${CMAKE_COMMAND} -E remove allocator_test.out)
set_tests_properties(allocator_test.fixture_cleanup PROPERTIES
  FIXTURES_CLEANUP allocator_test_output
)
find_program(VALGRIND_COMMAND valgrind)
if (VALGRIND_COMMAND)
  add_test(allocator_memcheck.valgrind
   ${VALGRIND_COMMAND} --trace-children=yes --leak-check=full ./allocator_test --arraysize 5000000 --passes 200)
  set_tests_properties(allocator_memcheck.valgrind PROPERTIES
    PASS_REGULAR_EXPRESSION "ERROR SUMMARY: 0 errors from 0 contexts"
  )
endif()

# Concurrency tests
add_test(allocator_concurrency_test.run allocator_hpx_test -t4 --passes 20 --outputfile allocator_concurrency_test.out)
set_tests_properties(allocator_concurrency_test.run PROPERTIES
  FIXTURES_SETUP allocator_concurrency_output
  PROCESSORS 4
)
add_test(allocator_concurrency_test.analyse_recycle_rate cat allocator_concurrency_test.out)
set_tests_properties(allocator_concurrency_test.analyse_recycle_rate PROPERTIES
  FIXTURES_REQUIRED allocator_concurrency_output
  PASS_REGULAR_EXPRESSION "==> Recycle rate: [ ]* 99.844%"
)
add_test(allocator_concurrency_test.analyse_marked_buffers_cleanup cat allocator_concurrency_test.out)
set_tests_properties(allocator_concurrency_test.analyse_marked_buffers_cleanup PROPERTIES
  FIXTURES_REQUIRED allocator_concurrency_output
  PASS_REGULAR_EXPRESSION "--> Number of buffers that were marked as used upon cleanup:[ ]* 0"
)
add_test(allocator_concurrency_test.analyse_cleaned_buffers cat allocator_concurrency_test.out)
set_tests_properties(allocator_concurrency_test.analyse_cleaned_buffers PROPERTIES
  FIXTURES_REQUIRED allocator_concurrency_output
  PASS_REGULAR_EXPRESSION "--> Number cleaned up buffers:[ ]* 4"
)
add_test(allocator_concurrency_test.analyse_created_buffers cat allocator_concurrency_test.out)
set_tests_properties(allocator_concurrency_test.analyse_created_buffers PROPERTIES
  FIXTURES_REQUIRED allocator_concurrency_output
  PASS_REGULAR_EXPRESSION "--> Number of times a new buffer had to be created for a request:[ ]* 4"
)
add_test(allocator_concurrency_test.analyse_bad_allocs cat allocator_concurrency_test.out)
set_tests_properties(allocator_concurrency_test.analyse_bad_allocs PROPERTIES
  FIXTURES_REQUIRED allocator_concurrency_output
  PASS_REGULAR_EXPRESSION "--> Number of bad_allocs that triggered garbage collection: [ ]* 0"
)
if (NOT CMAKE_BUILD_TYPE MATCHES "Debug") # Performance tests only make sense with optimizations on
  add_test(allocator_concurrency_test.performance.analyse_recycle_performance cat allocator_concurrency_test.out)
  set_tests_properties(allocator_concurrency_test.performance.analyse_recycle_performance PROPERTIES
    FIXTURES_REQUIRED allocator_concurrency_output
    PASS_REGULAR_EXPRESSION "Test information: Recycler was faster than default allocator!"
  )
  add_test(allocator_concurrency_test.performance.analyse_aggressive_performance cat allocator_concurrency_test.out)
  set_tests_properties(allocator_concurrency_test.performance.analyse_aggressive_performance PROPERTIES
    FIXTURES_REQUIRED allocator_concurrency_output
    PASS_REGULAR_EXPRESSION "Test information: Recycler was faster than default allocator!"
  )
endif()
add_test(allocator_concurrency_test.fixture_cleanup ${CMAKE_COMMAND} -E remove allocator_concurrency_test.out)
set_tests_properties(allocator_concurrency_test.fixture_cleanup PROPERTIES
  FIXTURES_CLEANUP allocator_concurrency_output
)


# Aligned alloc tests
add_test(allocator_aligned_test.run allocator_aligned_test --arraysize 5000000 --passes 200 --outputfile allocator_aligned_test.out)
set_tests_properties(allocator_aligned_test.run PROPERTIES
  FIXTURES_SETUP allocator_aligned_test_output
)
add_test(allocator_aligned_test.analyse_recycle_rate cat allocator_aligned_test.out)
set_tests_properties(allocator_aligned_test.analyse_recycle_rate PROPERTIES
  FIXTURES_REQUIRED allocator_aligned_test_output
  PASS_REGULAR_EXPRESSION "==> Recycle rate: [ ]* 99.5%"
)
add_test(allocator_aligned_test.analyse_marked_buffers_cleanup cat allocator_aligned_test.out)
set_tests_properties(allocator_aligned_test.analyse_marked_buffers_cleanup PROPERTIES
  FIXTURES_REQUIRED allocator_aligned_test_output
  PASS_REGULAR_EXPRESSION "--> Number of buffers that were marked as used upon cleanup:[ ]* 0"
)
add_test(allocator_aligned_test.analyse_cleaned_buffers cat allocator_aligned_test.out)
set_tests_properties(allocator_aligned_test.analyse_cleaned_buffers PROPERTIES
  FIXTURES_REQUIRED allocator_aligned_test_output
  PASS_REGULAR_EXPRESSION "--> Number cleaned up buffers:[ ]* 1"
)
add_test(allocator_aligned_test.analyse_created_buffers cat allocator_aligned_test.out)
set_tests_properties(allocator_aligned_test.analyse_created_buffers PROPERTIES
  FIXTURES_REQUIRED allocator_aligned_test_output
  PASS_REGULAR_EXPRESSION "--> Number of times a new buffer had to be created for a request:[ ]* 1"
)
add_test(allocator_aligned_test.analyse_bad_allocs cat allocator_aligned_test.out)
set_tests_properties(allocator_aligned_test.analyse_bad_allocs PROPERTIES
  FIXTURES_REQUIRED allocator_aligned_test_output
  PASS_REGULAR_EXPRESSION "--> Number of bad_allocs that triggered garbage collection: [ ]* 0"
)
if (NOT CMAKE_BUILD_TYPE MATCHES "Debug") # Performance tests only make sense with optimizations on
  add_test(allocator_aligned_test.performance.analyse_recycle_performance cat allocator_aligned_test.out)
  set_tests_properties(allocator_aligned_test.performance.analyse_recycle_performance PROPERTIES
    FIXTURES_REQUIRED allocator_aligned_test_output
    PASS_REGULAR_EXPRESSION "Test information: Recycler was faster than default allocator!"
  )
  add_test(allocator_aligned_test.performance.analyse_aggressive_performance cat allocator_aligned_test.out)
  set_tests_properties(allocator_aligned_test.performance.analyse_aggressive_performance PROPERTIES
    FIXTURES_REQUIRED allocator_aligned_test_output
    PASS_REGULAR_EXPRESSION "Test information: Recycler was faster than default allocator!"
  )
endif()
add_test(allocator_aligned_test.fixture_cleanup ${CMAKE_COMMAND} -E remove allocator_aligned_test.out)
set_tests_properties(allocator_aligned_test.fixture_cleanup PROPERTIES
  FIXTURES_CLEANUP allocator_aligned_test_output
)
find_program(VALGRIND_COMMAND valgrind)
if (VALGRIND_COMMAND)
  add_test(allocator_memcheck.valgrind
   ${VALGRIND_COMMAND} --trace-children=yes --leak-check=full ./allocator_aligned_test --arraysize 5000000 --passes 200)
  set_tests_properties(allocator_memcheck.valgrind PROPERTIES
    PASS_REGULAR_EXPRESSION "ERROR SUMMARY: 0 errors from 0 contexts"
  )
endif()

# GPU related tests
if (INTEROPT_WITH_CUDA)
  add_test(allocator_cuda_test.run allocator_cuda_test)
  add_test(allocator_kokkos_minimal.run minimal_parallel_for)
  #add_test(allocator_executor_stream_test.run allocator_executor_stream_test)
  add_test(allocator_kokkos_executor_stream_test.run allocator_kokkos_executor_stream_test)
  add_test(stream_test.run stream_test)
endif()

endif()
