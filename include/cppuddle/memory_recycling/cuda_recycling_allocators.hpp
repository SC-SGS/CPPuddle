// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_RECYCLING_ALLOCATORS_HPP
#define CUDA_RECYCLING_ALLOCATORS_HPP

#include "buffer_management_interface.hpp"
// import cuda_pinned_allocator and cuda_device_allocator
#include "detail/cuda_underlying_allocators.hpp"

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

} // namespace memory_recycling
} // end namespace cppuddle
#endif
