#================================================================================================================================================
# General Setup
#================================================================================================================================================
cmake_minimum_required(VERSION 3.0.2)

include(FindPackageHandleStandardArgs)

include(ProcessorCount)
ProcessorCount(N)
if (N EQUAL 0)
	set(N 1)
	MESSAGE(WARNING "Could not determine number of CPU cores! Submodules will be built with -j 1")
endif()

project(hpx_kokkos_example)
#================================================================================================================================================
# Setup HPX
#================================================================================================================================================
# Check whether the Hpx module is initialised and issue error if not
find_path(HPX_DIR NAMES "CMakeLists.txt" PATHS "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx")
if ((NOT HPX_DIR) OR (NOT EXISTS ${HPX_DIR}))
	MESSAGE(FATAL_ERROR "Could not find the CMakeLists file within the Hpx directory!\n"
		            "To fix this error, clone to Hpx submodule by running the following command within the project root directory:\n"
			        "git submodule update --init external_dependencies/hpx")
endif()

#-DHpx_ARCH_SKX=ON -DHpx_ARCH_PASCAL60=ON -DHpx_ENABLE_HPX=1
#-DHpx_ENABLE_SERIAL=1 -DHpx_ENABLE_CUDA=1 -DHpx_ENABLE_CUDA_LAMBDA=1
#-DHPX_DIR=$...

execute_process(COMMAND "mkdir" "-p" "hpx-build/build" 
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies")

# Run Hpx CMakE
execute_process(COMMAND "cmake" "=DHPX_WITH_CUDA=ON" "-B=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build"
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build" "-DHPX_WITH_EXAMPLES=OFF"
  "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build")
# Run make
execute_process(COMMAND "cmake" "--build" "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build" "--" "-j${N}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build")
execute_process(COMMAND "cmake" "--build" "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build" "--target" "install"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build")

set(HPX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build)

include_directories(${HPX_DIR}/include)

link_directories(${HPX_DIR}/lib)

#================================================================================================================================================
# Setup Kokkos
#================================================================================================================================================
# Check whether the Kokkos module is initialised and issue error if not
find_path(KOKKOS_DIR NAMES "CMakeLists.txt" PATHS "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos")
if ((NOT KOKKOS_DIR) OR (NOT EXISTS ${KOKKOS_DIR}))
	MESSAGE(FATAL_ERROR "Could not find the CMakeLists file within the Kokkos directory!\n"
		            "To fix this error, clone to Kokkos submodule by running the following command within the project root directory:\n"
			        "git submodule update --init external_dependencies/kokkos")
endif()

execute_process(COMMAND "mkdir" "-p" "kokkos-build/install" 
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies")

# Run Kokkos CMakE
execute_process(COMMAND "cmake" "-DKOKKOS_ENABLE_OPENMP=1" "-DKOKKOS_ENABLE_CUDA=1"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos/bin/nvcc_wrapper"
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-build/install"
  "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-build")
execute_process(COMMAND "make" "-j${N}" "VERBOSE=1"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-build")
execute_process(COMMAND "make" "install"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-build")

#set(KOKKOS_DIR /home/daissgr/workshop/kokkos-spielwiese/kokkos2/build/install)
#set(CMAKE_CXX_COMPILER /home/daissgr/workshop/kokkos-spielwiese/kokkos2/bin/nvcc_wrapper)

set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos/bin/nvcc_wrapper)
set(KOKKOS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-build/install)

enable_language(CUDA)
add_compile_options(-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored) # surpresses annotation warning, see https://github.com/kokkos/kokkos/issues/1473 

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

include_directories(${KOKKOS_DIR}/include)

link_directories(${KOKKOS_DIR}/lib)


#================================================================================================================================================
# Setup application
#================================================================================================================================================
link_libraries(kokkos hpx dl boost_system boost_program_options boost_thread boost_regex boost_chrono boost_atomic boost_filesystem)


# Headers of the current application
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Sources of the current application
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/hello.cpp)

# We need C++11 for our example
add_compile_options(-std=c++14) 

add_executable(hello ${SOURCES})
