# Copyright (c) 2020-2021 Gregor Daiß

# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.11)

project(CPPuddle CXX C) # Some random project name
set(CMAKE_CXX_STANDARD 17)

option(CPPUDDLE_WITH_TESTS "Build tests/examples" OFF)
set(CPPUDDLE_WITH_DEADLOCK_TEST_REPETITONS "100000" CACHE STRING "Number of repetitions for the aggregation executor deadlock tests")
option(CPPUDDLE_WITH_COUNTERS "Turns on allocations counters. Useful for extended testing" OFF)
option(CPPUDDLE_WITH_CUDA "Enable CUDA tests/examples" OFF)
option(CPPUDDLE_WITH_MULTIGPU_SUPPORT "Enables experimental MultiGPU support" OFF)
option(CPPUDDLE_WITH_HPX "Enable HPX examples" OFF)
option(CPPUDDLE_WITH_KOKKOS "Enable KOKKOS tests/examples" OFF)
option(CPPUDDLE_WITH_CLANG_TIDY "Enable clang tidy warnings" OFF)
option(CPPUDDLE_WITH_CLANG_FORMAT "Enable clang format target" OFF)

if (CPPUDDLE_WITH_CUDA)
   enable_language(CUDA) 
endif ()

if (CPPUDDLE_WITH_HPX)
  find_package(HPX REQUIRED)
endif()
if (CPPUDDLE_WITH_TESTS)
  find_package(Boost REQUIRED program_options)
endif()
if (CPPUDDLE_WITH_KOKKOS)
  # Find packages
  find_package(Kokkos 3.0.0 REQUIRED)
  find_package(HPXKokkos REQUIRED) 

  # Check that everything required is actyivated
  if (NOT CPPUDDLE_WITH_HPX)
    message(FATAL_ERROR "KOKKOS support requires HPX flag to be turned on")
  endif()
  #if (NOT CPPUDDLE_WITH_CUDA AND NOT CPPUDDLE_WITH_HIP)
  #  message(FATAL_ERROR "KOKKOS support requires CUDA flag to be turned on")
  #endif()

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
endif()

# Add Linter warnings
if (CPPUDDLE_WITH_CLANG_TIDY)
  find_program(CLANG_TIDY "clang-tidy")
  if(CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-header-filter=.;-checks=*,-fuchsia*,-misc-unused-parameters,-readability-named-parameter,-cert*,-android*,-modernize-pass-by-value,-hicpp-use-equals-default,-modernize-use-equals-default")
  endif()
endif()

# Define clang format target
if (CPPUDDLE_WITH_CLANG_FORMAT)
  file(GLOB_RECURSE
      ALL_CXX_SOURCE_FILES
      src/*.cpp tests/*.cpp test/*.cu tests/*.hpp src/*.hpp include/*.hpp src/*.cu
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
endif()


## Interface targets
add_library(buffer_manager SHARED src/buffer_manager_definitions.cpp)
if (CPPUDDLE_WITH_HPX)
  target_compile_definitions(buffer_manager INTERFACE "CPPUDDLE_HAVE_HPX")
endif()
if (CPPUDDLE_WITH_COUNTERS)
  target_compile_definitions(buffer_manager INTERFACE "CPPUDDLE_HAVE_COUNTERS")
endif()
target_include_directories(buffer_manager INTERFACE
 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
 $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
 )

add_library(stream_manager SHARED src/stream_manager_definitions.cpp)
target_link_libraries(stream_manager
    PRIVATE )
target_include_directories(stream_manager INTERFACE 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include> 
)

# install libs with the defitions:
install(TARGETS buffer_manager EXPORT CPPuddle
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib 
)
install(TARGETS stream_manager EXPORT CPPuddle
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib  
)
# install all headers
install(
  DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

install(FILES cppuddle-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/CPPuddle/)
install(EXPORT CPPuddle NAMESPACE CPPuddle:: DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/CPPuddle/)

## Add target for tests and tests definitions
if (CPPUDDLE_WITH_TESTS)
  add_executable(allocator_test tests/allocator_test.cpp)
  target_link_libraries(allocator_test
  ${Boost_LIBRARIES} Boost::boost Boost::program_options buffer_manager)

  add_executable(allocator_aligned_test tests/allocator_aligned_test.cpp)
  target_link_libraries(allocator_aligned_test
  ${Boost_LIBRARIES} Boost::boost Boost::program_options buffer_manager)

  if (CPPUDDLE_WITH_HPX)

    add_executable(allocator_hpx_test tests/allocator_hpx_test.cpp)
    target_link_libraries(allocator_hpx_test
      PRIVATE Boost::boost Boost::program_options HPX::hpx buffer_manager)

    if (CPPUDDLE_WITH_CUDA)

      add_executable(allocator_cuda_test tests/allocator_cuda_test.cu)
      target_link_libraries(allocator_cuda_test
        PRIVATE Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
      target_compile_definitions(allocator_cuda_test PRIVATE CPPUDDLE_HAVE_CUDA)

      add_executable(
        stream_test
        tests/stream_test.cpp
        include/stream_manager.hpp)
      target_link_libraries(stream_test
        PRIVATE HPX::hpx buffer_manager stream_manager)
      target_compile_definitions(stream_test PRIVATE CPPUDDLE_HAVE_CUDA)

      if (CPPUDDLE_WITH_KOKKOS)

        add_executable(
          allocator_kokkos_test
          tests/allocator_kokkos_test.cpp
          include/buffer_manager.hpp)
        target_link_libraries(allocator_kokkos_test
          PRIVATE Boost::boost Boost::program_options HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
        target_compile_definitions(allocator_kokkos_test PRIVATE HPX_WITH_CUDA CPPUDDLE_HAVE_CUDA)

        add_hpx_executable(
          allocator_kokkos_executor_for_loop_test
          DEPENDENCIES
          HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager
          SOURCES
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/allocator_kokkos_executor_for_loop_test.cpp)
       # target_link_libraries(allocator_kokkos_executor_for_loop_test
       #   PRIVATE HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager)
        target_compile_definitions(allocator_kokkos_executor_for_loop_test PRIVATE HPX_WITH_CUDA CPPUDDLE_HAVE_CUDA)

        add_hpx_executable(
          work_aggregation_test
          DEPENDENCIES
          Boost::boost Boost::program_options HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager stream_manager
          COMPONENT_DEPENDENCIES iostreams
          SOURCES
          tests/work_aggregation_test.cpp
          include/aggregation_manager.hpp
          include/buffer_manager.hpp
          include/stream_manager.hpp
          )

        add_hpx_executable(
          work_aggregation_cpu_triad
          DEPENDENCIES
          Boost::boost Boost::program_options HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager stream_manager
          COMPONENT_DEPENDENCIES iostreams
          SOURCES
          tests/work_aggregation_cpu_triad.cpp
          include/aggregation_manager.hpp
          include/buffer_manager.hpp
          include/stream_manager.hpp
          )

        add_hpx_executable(
          work_aggregation_cuda_triad
          DEPENDENCIES
          Boost::boost Boost::program_options HPX::hpx Kokkos::kokkos HPXKokkos::hpx_kokkos buffer_manager stream_manager
          COMPONENT_DEPENDENCIES iostreams
          SOURCES
          tests/work_aggregation_cuda_triad.cpp
          include/aggregation_manager.hpp
          include/buffer_manager.hpp
          include/stream_manager.hpp
          )
        target_compile_definitions(work_aggregation_test PRIVATE CPPUDDLE_HAVE_CUDA)
      endif() # end WITH KOKKOS
    endif() # end with CUDA
  endif() # end with HPX

  if (CPPUDDLE_WITH_COUNTERS)
    add_compile_definitions(CPPUDDLE_HAVE_COUNTERS)
  endif()
  if (CPPUDDLE_WITH_HPX)
    add_compile_definitions(CPPUDDLE_WITH_HPX)
  endif()

  if (CPPUDDLE_WITH_MULTIGPU_SUPPORT)
    add_compile_definitions(CPPUDDLE_HAVE_MULTIGPU)
    message(WARNING, " Multi-GPU Support not yet properly tested!")
  endif()

  enable_testing()

  # Basic functionality tests
  add_test(allocator_test.run allocator_test --arraysize 5000000 --passes 200 --outputfile allocator_test.out)
  set_tests_properties(allocator_test.run PROPERTIES
    FIXTURES_SETUP allocator_test_output
  )
  if (CPPUDDLE_WITH_COUNTERS)
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
  endif()
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
  
  # Aligned alloc tests
  add_test(allocator_aligned_test.run allocator_aligned_test --arraysize 5000000 --passes 200 --outputfile allocator_aligned_test.out)
  set_tests_properties(allocator_aligned_test.run PROPERTIES
    FIXTURES_SETUP allocator_aligned_test_output
  )
  if (CPPUDDLE_WITH_COUNTERS)
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
  endif()
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

  if (CPPUDDLE_WITH_HPX)
    # Concurrency tests
    add_test(allocator_concurrency_test.run allocator_hpx_test -t4 --passes 20 --outputfile allocator_concurrency_test.out)
    set_tests_properties(allocator_concurrency_test.run PROPERTIES
      FIXTURES_SETUP allocator_concurrency_output
      PROCESSORS 4
    )
    if (CPPUDDLE_WITH_COUNTERS)
      add_test(allocator_concurrency_test.analyse_recycle_rate cat allocator_concurrency_test.out)
      set_tests_properties(allocator_concurrency_test.analyse_recycle_rate PROPERTIES
        FIXTURES_REQUIRED allocator_concurrency_output
        PASS_REGULAR_EXPRESSION "==> Recycle rate: [ ]* 99.6885%"
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
    endif()
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

    # GPU related tests
    if (CPPUDDLE_WITH_CUDA)
      add_test(allocator_cuda_test.run allocator_cuda_test -t 4)
      set_tests_properties(allocator_cuda_test.run PROPERTIES
        PROCESSORS 4
      )
      add_test(stream_test.run stream_test -t 4)
      set_tests_properties(stream_test.run PROPERTIES
        PROCESSORS 4
      )
      if (CPPUDDLE_WITH_KOKKOS)
        add_test(allocator_kokkos_test.run allocator_kokkos_test --outputfile allocator_kokkos_test.out)
        set_tests_properties(allocator_kokkos_test.run PROPERTIES
          FIXTURES_SETUP allocator_kokkos_output
        )

        add_test(allocator_kokkos_test.analyse_recycle_rate cat allocator_kokkos_test.out)
        set_tests_properties(allocator_kokkos_test.analyse_recycle_rate PROPERTIES
          FIXTURES_REQUIRED allocator_kokkos_output
          PASS_REGULAR_EXPRESSION "==> Recycle rate: [ ]* 99%"
        )
        add_test(allocator_kokkos_test.analyse_marked_buffers_cleanup cat allocator_kokkos_test.out)
        set_tests_properties(allocator_kokkos_test.analyse_marked_buffers_cleanup PROPERTIES
          FIXTURES_REQUIRED allocator_kokkos_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that were marked as used upon cleanup:[ ]* 0"
        )
        add_test(allocator_kokkos_test.analyse_cleaned_buffers cat allocator_kokkos_test.out)
        set_tests_properties(allocator_kokkos_test.analyse_cleaned_buffers PROPERTIES
          FIXTURES_REQUIRED allocator_kokkos_output
          PASS_REGULAR_EXPRESSION "--> Number cleaned up buffers:[ ]* 2"
        )
        add_test(allocator_kokkos_test.analyse_created_buffers cat allocator_kokkos_test.out)
        set_tests_properties(allocator_kokkos_test.analyse_created_buffers PROPERTIES
          FIXTURES_REQUIRED allocator_kokkos_output
          PASS_REGULAR_EXPRESSION "--> Number of times a new buffer had to be created for a request:[ ]* 2"
        )
        add_test(allocator_kokkos_test.analyse_bad_allocs cat allocator_kokkos_test.out)
        set_tests_properties(allocator_kokkos_test.analyse_bad_allocs PROPERTIES
          FIXTURES_REQUIRED allocator_kokkos_output
          PASS_REGULAR_EXPRESSION "--> Number of bad_allocs that triggered garbage collection: [ ]* 0"
        )
        add_test(allocator_kokkos_executor_for_loop_test.run allocator_kokkos_executor_for_loop_test -t 4)
        set_tests_properties(allocator_kokkos_executor_for_loop_test.run PROPERTIES
          PROCESSORS 4
        )

        add_test(aggregation_basic_sequential_test.run work_aggregation_test -t1 --outputfile=aggregation_basic_sequential_test.out --scenario=sequential_test)
        set_tests_properties(aggregation_basic_sequential_test.run PROPERTIES
          FIXTURES_SETUP aggregation_basic_sequential_test_output
        )
        add_test(aggregation_basic_sequential_test.analyse_int_buffers cat aggregation_basic_sequential_test.out)
        set_tests_properties(aggregation_basic_sequential_test.analyse_int_buffers PROPERTIES
          FIXTURES_REQUIRED aggregation_basic_sequential_test_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that got requested from this manager: [ ]* 2"
        )
        add_test(aggregation_basic_sequential_test.analyse_float_buffers cat aggregation_basic_sequential_test.out)
        set_tests_properties(aggregation_basic_sequential_test.analyse_float_buffers PROPERTIES
          FIXTURES_REQUIRED aggregation_basic_sequential_test_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that got requested from this manager: [ ]* 6"
        )
        add_test(aggregation_basic_sequential_test.analyse_cleanup cat aggregation_basic_sequential_test.out)
        set_tests_properties(aggregation_basic_sequential_test.analyse_cleanup PROPERTIES
          FIXTURES_REQUIRED aggregation_basic_sequential_test_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that were marked as used upon cleanup: [ ]* 0"
        )

        add_test(aggregation_basic_parallel_test.run work_aggregation_test -t4 --outputfile=aggregation_basic_parallel_test.out --scenario=sequential_test)
        set_tests_properties(aggregation_basic_parallel_test.run PROPERTIES
          FIXTURES_SETUP aggregation_basic_parallel_test_output
          PROCESSORS 4
        )
        add_test(aggregation_basic_parallel_test.analyse_int_buffers cat aggregation_basic_parallel_test.out)
        set_tests_properties(aggregation_basic_parallel_test.analyse_int_buffers PROPERTIES
          FIXTURES_REQUIRED aggregation_basic_parallel_test_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that got requested from this manager: [ ]* 2"
        )
        add_test(aggregation_basic_parallel_test.analyse_float_buffers cat aggregation_basic_parallel_test.out)
        set_tests_properties(aggregation_basic_parallel_test.analyse_float_buffers PROPERTIES
          FIXTURES_REQUIRED aggregation_basic_parallel_test_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that got requested from this manager: [ ]* 6"
        )
        add_test(aggregation_basic_parallel_test.analyse_cleanup cat aggregation_basic_parallel_test.out)
        set_tests_properties(aggregation_basic_parallel_test.analyse_cleanup PROPERTIES
          FIXTURES_REQUIRED aggregation_basic_parallel_test_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that were marked as used upon cleanup: [ ]* 0"
        )



        add_test(aggregation_interruption_test.run work_aggregation_test -t1 --outputfile=aggregation_interruption_test.out --scenario=interruption_test)
        set_tests_properties(aggregation_interruption_test.run PROPERTIES
          FIXTURES_SETUP aggregation_interruption_test_output
        )
        add_test(aggregation_interruption_test.analyse_continuation_output cat aggregation_interruption_test.out)
        set_tests_properties(aggregation_interruption_test.analyse_continuation_output PROPERTIES
          FIXTURES_REQUIRED aggregation_interruption_test_output
          PASS_REGULAR_EXPRESSION "Got executor 1"
        )
        add_test(aggregation_interruption_test.analyse_kernel_output cat aggregation_interruption_test.out)
        set_tests_properties(aggregation_interruption_test.analyse_kernel_output PROPERTIES
          FIXTURES_REQUIRED aggregation_interruption_test_output
          PASS_REGULAR_EXPRESSION "d is 2"
        )



        add_test(aggregation_failure_test.run work_aggregation_test -t1 --outputfile=aggregation_failure_test.out --scenario=failure_test)
        set_tests_properties(aggregation_failure_test.run PROPERTIES
          FIXTURES_SETUP aggregation_failure_test_output
        )
        add_test(aggregation_failure_test.analyse_value_error cat aggregation_failure_test.out)
        set_tests_properties(aggregation_failure_test.analyse_value_error PROPERTIES
          FIXTURES_REQUIRED aggregation_failure_test_output
          PASS_REGULAR_EXPRESSION "Mismatched values error in aggregated post call of executor"
        )
        add_test(aggregation_failure_test.analyse_type_error cat aggregation_failure_test.out)
        set_tests_properties(aggregation_failure_test.analyse_type_error PROPERTIES
          FIXTURES_REQUIRED aggregation_failure_test_output
          PASS_REGULAR_EXPRESSION "Mismatched types error in aggregated post call of executor"
        )


        add_test(aggregation_add_pointer_test.run work_aggregation_test -t4 --outputfile=aggregation_add_pointer_test.out --scenario=pointer_add_test)
        set_tests_properties(aggregation_add_pointer_test.run PROPERTIES
          FIXTURES_SETUP aggregation_add_pointer_test_output
          PROCESSORS 4
        )
        add_test(aggregation_add_pointer_test.analyse_number_launches cat aggregation_add_pointer_test.out)
        set_tests_properties(aggregation_add_pointer_test.analyse_number_launches PROPERTIES
          FIXTURES_REQUIRED aggregation_add_pointer_test_output
          PASS_REGULAR_EXPRESSION "Number add_pointer_launches=2"
        )
        add_test(aggregation_add_pointer_test.analyse_number_buffers cat aggregation_add_pointer_test.out)
        set_tests_properties(aggregation_add_pointer_test.analyse_number_buffers PROPERTIES
          FIXTURES_REQUIRED aggregation_add_pointer_test_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that got requested from this manager: [ ]* 6"
        )

        add_test(aggregation_add_references_test.run work_aggregation_test -t4 --outputfile=aggregation_add_references_test.out --scenario=references_add_test)
        set_tests_properties(aggregation_add_references_test.run PROPERTIES
          FIXTURES_SETUP aggregation_add_references_test_output
          PROCESSORS 4
        )
        add_test(aggregation_add_references_test.analyse_number_launches cat aggregation_add_references_test.out)
        set_tests_properties(aggregation_add_references_test.analyse_number_launches PROPERTIES
          FIXTURES_REQUIRED aggregation_add_references_test_output
          PASS_REGULAR_EXPRESSION "Number add_launches=1"
        )
        add_test(aggregation_add_references_test.analyse_number_buffers cat aggregation_add_references_test.out)
        set_tests_properties(aggregation_add_references_test.analyse_number_buffers PROPERTIES
          FIXTURES_REQUIRED aggregation_add_references_test_output
          PASS_REGULAR_EXPRESSION "--> Number of buffers that got requested from this manager: [ ]* 3"
        )


      # STREAM TESTS CPU

      # set to larger value for Jenkins checks?
      set(deadlock_check_repetitions ${CPPUDDLE_WITH_DEADLOCK_TEST_REPETITONS})
      #set(deadlock_check_repetitions "100000")
      message(STATUS "Deadlock check repetitions set to ${deadlock_check_repetitions}")

      # Try with few slices -- good to detect deadlocking on errors with the continuations
        add_test(aggregation_stream_triad_cpu_eager_test1.run work_aggregation_cpu_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=2048 --problem_size=25600 --kernel_size=256 --max_slices=2 --repetitions=${deadlock_check_repetitions} --executor_type=EAGER --outputfile=aggregation_stream_triad_cpu_eager_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test1.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cpu_eager_test_output1
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cpu_eager_test1.check_errors cat aggregation_stream_triad_cpu_eager_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test1.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_eager_test_output1
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cpu_eager_test1.check_success cat aggregation_stream_triad_cpu_eager_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test1.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_eager_test_output1
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # Try with odd number of slices
      # This would deadlock given the STRICT executor, the EAGER one should have no problem
        add_test(aggregation_stream_triad_cpu_eager_test2.run work_aggregation_cpu_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=2048 --problem_size=25600 --kernel_size=256 --max_slices=17 --repetitions=${deadlock_check_repetitions} --executor_type=EAGER --outputfile=aggregation_stream_triad_cpu_eager_test2.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test2.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cpu_eager_test_output2
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cpu_eager_test2.check_errors cat aggregation_stream_triad_cpu_eager_test2.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test2.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_eager_test_output2
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cpu_eager_test2.check_success cat aggregation_stream_triad_cpu_eager_test2.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test2.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_eager_test_output2
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # Try with large number of slices -- this is basically what should be used in production, hence it should be tested
        add_test(aggregation_stream_triad_cpu_eager_test3.run work_aggregation_cpu_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=2048 --problem_size=25600 --kernel_size=256 --max_slices=100 --repetitions=${deadlock_check_repetitions} --executor_type=EAGER --outputfile=aggregation_stream_triad_cpu_eager_test3.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test3.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cpu_eager_test_output3
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cpu_eager_test3.check_errors cat aggregation_stream_triad_cpu_eager_test3.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test3.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_eager_test_output3
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cpu_eager_test3.check_success cat aggregation_stream_triad_cpu_eager_test3.out)
        set_tests_properties(aggregation_stream_triad_cpu_eager_test3.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_eager_test_output3
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # Basic test for the ENDLESS executor -- number slices should not matter here, hence the large value for it
      add_test(aggregation_stream_triad_cpu_endless_test1.run work_aggregation_cpu_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=2048 --problem_size=25600 --kernel_size=256 --max_slices=99999999 --repetitions=${deadlock_check_repetitions} --executor_type=ENDLESS --outputfile=aggregation_stream_triad_cpu_endless_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_endless_test1.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cpu_endless_test_output1
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cpu_endless_test1.check_errors cat aggregation_stream_triad_cpu_endless_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_endless_test1.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_endless_test_output1
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cpu_endless_test1.check_success cat aggregation_stream_triad_cpu_endless_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_endless_test1.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_endless_test_output1
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # Basic test for the STRICT executor 
      add_test(aggregation_stream_triad_cpu_strict_test1.run work_aggregation_cpu_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=2048 --problem_size=25600 --kernel_size=256 --max_slices=100 --repetitions=${deadlock_check_repetitions} --executor_type=STRICT --outputfile=aggregation_stream_triad_cpu_strict_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_test1.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cpu_strict_test_output1
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cpu_strict_test1.check_errors cat aggregation_stream_triad_cpu_strict_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_test1.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_test_output1
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cpu_strict_test1.check_success cat aggregation_stream_triad_cpu_strict_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_test1.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_test_output1
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # STRICT number of kernel launches should always be same -- hence we can check the aggregation working correctly here -- here it should be exactly 200 (no aggregation happening)
      add_test(aggregation_stream_triad_cpu_strict_aggregation_test1.run work_aggregation_cpu_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=2048 --problem_size=25600 --kernel_size=256 --max_slices=1 --repetitions=2 --executor_type=STRICT --outputfile=aggregation_stream_triad_cpu_strict_aggregation_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test1.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cpu_strict_aggregation_test_output1
          PROCESSORS 4
          TIMEOUT 90
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test1.check_errors cat aggregation_stream_triad_cpu_strict_aggregation_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test1.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output1
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test1.check_success cat aggregation_stream_triad_cpu_strict_aggregation_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test1.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output1
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test1.check_launch_counter cat aggregation_stream_triad_cpu_strict_aggregation_test1.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test1.check_launch_counter PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output1
          PASS_REGULAR_EXPRESSION "Kernel launch counter: 200"
        )


      # STRICT number of kernel launches should always be same -- hence we can check the aggregation working correctly here -- here it should be exactly 30
      add_test(aggregation_stream_triad_cpu_strict_aggregation_test2.run work_aggregation_cpu_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=2048 --problem_size=25600 --kernel_size=256 --max_slices=10 --repetitions=3 --executor_type=STRICT --outputfile=aggregation_stream_triad_cpu_strict_aggregation_test2.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test2.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cpu_strict_aggregation_test_output2
          PROCESSORS 4
          TIMEOUT 90
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test2.check_errors cat aggregation_stream_triad_cpu_strict_aggregation_test2.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test2.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output2
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test2.check_success cat aggregation_stream_triad_cpu_strict_aggregation_test2.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test2.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output2
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test2.check_launch_counter cat aggregation_stream_triad_cpu_strict_aggregation_test2.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test2.check_launch_counter PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output2
          PASS_REGULAR_EXPRESSION "Kernel launch counter: 30"
        )


      # STRICT number of kernel launches should always be same -- hence we can check the aggregation working correctly here -- here it should be exactly 1
      add_test(aggregation_stream_triad_cpu_strict_aggregation_test3.run work_aggregation_cpu_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=2048 --problem_size=25600 --kernel_size=256 --max_slices=100 --repetitions=1 --executor_type=STRICT --outputfile=aggregation_stream_triad_cpu_strict_aggregation_test3.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test3.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cpu_strict_aggregation_test_output3
          PROCESSORS 4
          TIMEOUT 90
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test3.check_errors cat aggregation_stream_triad_cpu_strict_aggregation_test3.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test3.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output3
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test3.check_success cat aggregation_stream_triad_cpu_strict_aggregation_test3.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test3.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output3
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )
        add_test(aggregation_stream_triad_cpu_strict_aggregation_test3.check_launch_counter cat aggregation_stream_triad_cpu_strict_aggregation_test3.out)
        set_tests_properties(aggregation_stream_triad_cpu_strict_aggregation_test3.check_launch_counter PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cpu_strict_aggregation_test_output3
          PASS_REGULAR_EXPRESSION "Kernel launch counter: 1"
        )

      # STREAM TESTS CUDA

      # Try with few slices -- good to detect deadlocking on errors with the continuations
        add_test(aggregation_stream_triad_cuda_eager_test1.run work_aggregation_cuda_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=4 --problem_size=102400 --kernel_size=1024 --max_slices=2 --repetitions=${deadlock_check_repetitions} --executor_type=EAGER --outputfile=aggregation_stream_triad_cuda_eager_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test1.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cuda_eager_test_output1
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cuda_eager_test1.check_errors cat aggregation_stream_triad_cuda_eager_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test1.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_eager_test_output1
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cuda_eager_test1.check_success cat aggregation_stream_triad_cuda_eager_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test1.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_eager_test_output1
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # Try with odd number of slices
      # This would deadlock given the STRICT executor, the EAGER one should have no problem
        add_test(aggregation_stream_triad_cuda_eager_test2.run work_aggregation_cuda_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=4 --problem_size=102400 --kernel_size=1024 --max_slices=17 --repetitions=${deadlock_check_repetitions} --executor_type=EAGER --outputfile=aggregation_stream_triad_cuda_eager_test2.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test2.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cuda_eager_test_output2
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cuda_eager_test2.check_errors cat aggregation_stream_triad_cuda_eager_test2.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test2.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_eager_test_output2
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cuda_eager_test2.check_success cat aggregation_stream_triad_cuda_eager_test2.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test2.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_eager_test_output2
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # Try with large number of slices -- this is basically what should be used in production, hence it should be tested
        add_test(aggregation_stream_triad_cuda_eager_test3.run work_aggregation_cuda_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=4 --problem_size=102400 --kernel_size=1024 --max_slices=100 --repetitions=${deadlock_check_repetitions} --executor_type=EAGER --outputfile=aggregation_stream_triad_cuda_eager_test3.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test3.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cuda_eager_test_output3
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cuda_eager_test3.check_errors cat aggregation_stream_triad_cuda_eager_test3.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test3.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_eager_test_output3
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cuda_eager_test3.check_success cat aggregation_stream_triad_cuda_eager_test3.out)
        set_tests_properties(aggregation_stream_triad_cuda_eager_test3.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_eager_test_output3
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # Basic test for the ENDLESS executor -- number slices should not matter here, hence the large value for it
      add_test(aggregation_stream_triad_cuda_endless_test1.run work_aggregation_cuda_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=4 --problem_size=102400 --kernel_size=1024 --max_slices=99999999 --repetitions=${deadlock_check_repetitions} --executor_type=ENDLESS --outputfile=aggregation_stream_triad_cuda_endless_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_endless_test1.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cuda_endless_test_output1
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cuda_endless_test1.check_errors cat aggregation_stream_triad_cuda_endless_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_endless_test1.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_endless_test_output1
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cuda_endless_test1.check_success cat aggregation_stream_triad_cuda_endless_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_endless_test1.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_endless_test_output1
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      # Basic test for the STRICT executor 
      add_test(aggregation_stream_triad_cuda_strict_test1.run work_aggregation_cuda_triad -t 4 --number_aggregation_executors=1 --number_underlying_executors=4 --problem_size=102400 --kernel_size=1024 --max_slices=100 --repetitions=${deadlock_check_repetitions} --executor_type=STRICT --outputfile=aggregation_stream_triad_cuda_strict_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_strict_test1.run PROPERTIES
          FIXTURES_SETUP aggregation_stream_triad_cuda_strict_test_output1
          PROCESSORS 4
          TIMEOUT 600
        )
        add_test(aggregation_stream_triad_cuda_strict_test1.check_errors cat aggregation_stream_triad_cuda_strict_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_strict_test1.check_errors PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_strict_test_output1
          FAIL_REGULAR_EXPRESSION "ERROR"
        )
        add_test(aggregation_stream_triad_cuda_strict_test1.check_success cat aggregation_stream_triad_cuda_strict_test1.out)
        set_tests_properties(aggregation_stream_triad_cuda_strict_test1.check_success PROPERTIES
          FIXTURES_REQUIRED aggregation_stream_triad_cuda_strict_test_output1
          PASS_REGULAR_EXPRESSION "SUCCESS"
        )


      endif() # end with KOKKOS
    endif() # end with CUDA
  endif() # end with hpx
endif() # end with tests


