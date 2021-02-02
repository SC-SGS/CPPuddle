### CPPuddle

WARNING: This repo is work in progress and should not be relied on for production use!

#### Purpose

This repository was initially created to explore how to best use HPX and Kokkos together!
For fine-grained GPU tasks, we needed a way to avoid excessive allocations of one-usage GPU buffers (as allocations block the device for all streams) and creation/deletion of GPU executors (as those are usually tied to a stream which is expensive to create as well).

We currently test it in the experimental build of [Octo-Tiger](https://github.com/STEllAR-GROUP/octotiger), together with [HPX-Kokkos](https://github.com/STEllAR-GROUP/hpx-kokkos).

#### Tools provided by this

- Allocators that reuse previousely allocated buffers if available (works with normal heap memory, pinned memory, aligned memory, CUDA device memory, and Kokkos Views).
- Executor pools and various scheduling policies (round robin, priority queue, multi-gpu), which rely on reference counting to gauge the current load of a executor instead of querying the device itself.

#### Requirements

- c++14
- CMake (min 3.11)
- Optional (for the header-only utilities / test): CUDA, Boost, [HPX](https://github.com/STEllAR-GROUP/hpx), [Kokkos](https://github.com/kokkos/kokkos), [HPX-Kokkos](https://github.com/STEllAR-GROUP/hpx-kokkos)

The submodules can be used to obtain the optional dependencies! Otherwise the git submodules and additional buildscripts in scripts/ can safely be ignored!

#### Build / Install

```
  cmake -H/path/to/source -B$/path/to/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install/cppuddle -DCPPUDDLE_WITH_TESTS=OFF -DCPPUDDLE_WITH_COUNTERS=OFF                                                             
  cmake --build /path/to/build -- -j4 VERBOSE=1                                                                                                                                                                                                          
  cmake --build /path/to/build --target install  
```
If installed correctly, it can be used in other cmake-based projects via
```
find_package(CPPuddle REQUIRED)
```
