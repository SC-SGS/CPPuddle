// LANL Copyright Notice:
//
//© 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare. derivative works, distribute copies
// to the public, perform publicly and display publicly, and to permit others
// to do so.
//
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_RECYCLING_ALLOCATORS_HPP
#define CUDA_RECYCLING_ALLOCATORS_HPP

#include "buffer_management_interface.hpp"
// import underlying cuda allocators used here
#include "detail/cuda_underlying_allocators.hpp"

/// \file
/// Contains the CUDA recycling allocators (in the form of type aliases)
/// for both pinned host memory and device memory. Also contains the required
/// device selector for MultiGPU setups with these allocators.

namespace cppuddle {
namespace memory_recycling {

// Tell cppuddle how to select the device for the cuda allocators
namespace device_selection {
/// GPU device selector using the CUDA API for pinned host allocations
template <typename T>
struct select_device_functor<T, detail::cuda_pinned_allocator<T>> {
  void operator()(const size_t device_id) { cudaSetDevice(device_id); }
};
/// GPU selector using the CUDA API for pinned host allocations
template <typename T>
struct select_device_functor<T, detail::cuda_device_allocator<T>> {
  void operator()(const size_t device_id) { cudaSetDevice(device_id); }
};
} // namespace device_selection

/// Recycling allocator for CUDA pinned host memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_host =
    detail::aggressive_recycle_allocator<T, detail::cuda_pinned_allocator<T>>;
/// Recycling allocator for CUDA device memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_device =
    detail::recycle_allocator<T, detail::cuda_device_allocator<T>>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_managed =
    detail::aggressive_recycle_allocator<T, detail::cuda_managed_memory_allocator<T>>;

} // namespace memory_recycling
} // end namespace cppuddle
#endif
