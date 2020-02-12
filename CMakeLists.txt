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

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(SKIP_HPX_KOKKOS_BUILD False)

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

#message(FATAL_ERROR ${BOOST_ROOT})

if(NOT SKIP_HPX_KOKKOS_BUILD)
execute_process(COMMAND "mkdir" "-p" "hpx-build/build" 
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies")

# Run Hpx CMakE
execute_process(COMMAND "cmake" "-DCMAKE_CXX_STANDARD=14" "-DHPX_WITH_CUDA=ON" 
  	"-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-install" "-DHPX_WITH_EXAMPLES=OFF"
  	"${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx" 
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build")
# Run make
execute_process(COMMAND "cmake" "--build" "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build" "--" "-j${N}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build")
execute_process(COMMAND "cmake" "--build" "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build" "--target" "install"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build")
endif()

set(HPX_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-install)

set(HPX_DIR ${HPX_BASE_DIR}/lib/cmake/HPX/)
find_package(HPX REQUIRED)
message(INFO "HPX done:")

#================================================================================================================================================
# Setup Kokkos
#================================================================================================================================================
# Check whether the Kokkos module is initialised and issue error if not
	find_path(KOKKOS_DIR NAMES "CMakeLists.txt" PATHS "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master")
if ((NOT KOKKOS_DIR) OR (NOT EXISTS ${KOKKOS_DIR}))
	MESSAGE(FATAL_ERROR "Could not find the CMakeLists file within the Kokkos directory!\n"
			"To fix this error, clone to Kokkos submodule by running the following command within the project root directory:\n"
			"git submodule update --init external_dependencies/kokkos-master")
endif()

	execute_process(COMMAND "mkdir" "-p" "kokkos-master-build/install" 
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies")

if(NOT SKIP_HPX_KOKKOS_BUILD)
# Run Kokkos CMakE
	execute_process(COMMAND "cmake" "-DCMAKE_CXX_STANDARD=14" "-DKOKKOS_ENABLE_CUDA=1" "-DKOKKOS_ENABLE_CUDA_LAMBDA=1" "-DKOKKOS_ARCH=Pascal60"
				"-DJEMALLOC_ROOT=/apps/daint/UES/biddisco/gcc/7.3.0/jemalloc/5.2.1"
				"-DKokkos_ENABLE_SERIAL=1" "-DCMAKE_CXX_FLAGS=-isystem\ ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/include"
				"-DKokkos_ENABLE_HPX=1" "-DKokkos_HPX_DIR=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build/lib/cmake/HPX"
  "-DHPX_DIR=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/hpx-build/build/lib/cmake/HPX" "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master/bin/nvcc_wrapper"
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master-build/install"
  "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master-build")
execute_process(COMMAND "make" "-j${N}" "VERBOSE=1"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master-build")
execute_process(COMMAND "make" "install"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master-build")
endif()
#set(KOKKOS_DIR /home/daissgr/workshop/kokkos-master-spielwiese/kokkos-master2/build/install)
#set(CMAKE_CXX_COMPILER /home/daissgr/workshop/kokkos-master-spielwiese/kokkos-master2/bin/nvcc_wrapper)

set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master/bin/nvcc_wrapper)
set(KOKKOS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-master-build/install)

enable_language(CUDA)
# not sure what is introducing the std=c++11 flag for the kokkos project
add_compile_options(-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-extended-lambda) # suppresses annotation warning, see https://github.com/kokkos/kokkos/issues/1473 

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

include_directories(${KOKKOS_DIR}/include)

link_directories(${KOKKOS_DIR}/lib ${KOKKOS_DIR}/lib)

message(INFO "HPX done:")

#================================================================================================================================================
# Setup application
#================================================================================================================================================

link_directories(/usr/lib)

# Headers of the current application
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include, ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-hpx-interop/include)

# Sources of the current application
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/hello.cpp)

# We need C++14 for our example
add_compile_options(-std=c++14) 

link_libraries(kokkos hpx dl boost_system boost_program_options boost_thread boost_regex boost_chrono boost_atomic boost_filesystem)

add_hpx_executable(hello SOURCES ${SOURCES})

add_compile_options(-I${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-hpx-interop/include)
add_hpx_executable(interop SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/external_dependencies/kokkos-hpx-interop/src/kokkos_hpx.cpp)
