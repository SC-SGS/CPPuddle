### CPPuddle

WARNING: This repository is a work in progress and should not be relied on for production use!

[![ctest](https://github.com/SC-SGS/CPPuddle/actions/workflows/cmake.yml/badge.svg)](https://github.com/SC-SGS/CPPuddle/actions/workflows/cmake.yml)
[![Build Status](https://simsgs.informatik.uni-stuttgart.de/jenkins/buildStatus/icon?job=CPPuddle%2Fmaster&config=allbuilds)](https://simsgs.informatik.uni-stuttgart.de/jenkins/view/Octo-Tiger%20and%20Dependencies/job/CPPuddle/job/master/)


#### Purpose

This repository was initially created to explore how to best use HPX and Kokkos together!
For fine-grained GPU tasks, we needed a way to avoid excessive allocations of one-usage GPU buffers (as allocations block the device for all streams) and creation/deletion of GPU executors (as those are usually tied to a stream which is expensive to create as well).

We currently test it in the experimental build of [Octo-Tiger](https://github.com/STEllAR-GROUP/octotiger), together with [HPX-Kokkos](https://github.com/STEllAR-GROUP/hpx-kokkos).
In this use-case, allocating GPU buffers for all sub-grids in advance would have wasted a lot of memory. On the other hand, unified memory would have caused unnecessary GPU to CPU page migrations (as the old input data gets overwritten anyway). Allocating buffers on-the-fly would have blocked the device. Hence, we currently test this buffer management solution!

#### Tools provided by this repository

- Allocators that reuse previousely allocated buffers if available (works with normal heap memory, pinned memory, aligned memory, CUDA device memory, and Kokkos Views). Note that separate buffers do not coexist on a single chunk of continuous memory, but use different allocations. 
- Executor pools and various scheduling policies (round robin, priority queue, multi-gpu), which rely on reference counting to gauge the current load of a executor instead of querying the device itself.

#### Requirements

- C++14
- CMake (>= 3.11)
- Optional (for the header-only utilities / test): CUDA, Boost, [HPX](https://github.com/STEllAR-GROUP/hpx), [Kokkos](https://github.com/kokkos/kokkos), [HPX-Kokkos](https://github.com/STEllAR-GROUP/hpx-kokkos)

The submodules can be used to obtain the optional dependencies which are required for testing the header-only utilities. If these tests are not required, the submodule (and the respective buildscripts in /scripts) can be ignored safely.

#### Build / Install

```
  cmake -H/path/to/source -B$/path/to/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install/cppuddle -DCPPUDDLE_WITH_TESTS=OFF -DCPPUDDLE_WITH_COUNTERS=OFF                                                             
  cmake --build /path/to/build -- -j4 VERBOSE=1                                                                                                                                                                                                          
  cmake --build /path/to/build --target install  
```
If installed correctly, cppuddle can be used in other cmake-based projects via
```
find_package(CPPuddle REQUIRED)
```
