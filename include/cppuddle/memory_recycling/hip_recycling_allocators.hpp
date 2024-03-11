// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HIP_RECYCLING_ALLOCATORS_HPP
#define HIP_RECYCLING_ALLOCATORS_HPP

#include "buffer_management_interface.hpp"
// import hip_pinned_allocator and hip_device_allocator
#include "detail/hip_underlying_allocators.hpp"

/// \file
/// Contains the HIP recycling allocators (in the form of type aliases)
/// for both pinned host memory and device memory. Also contains the required
/// device selector for MultiGPU setups with these allocators.

namespace cppuddle {
namespace memory_recycling {

// Tell cppuddle how to select the device for the hip allocators
namespace device_selection {
/// GPU device selector using the HIP API for pinned host allocations
template <typename T>
struct select_device_functor<T, detail::hip_pinned_allocator<T>> {
  void operator()(const size_t device_id) { hipSetDevice(device_id); }
};
/// GPU selector using the HIP API for pinned host allocations
template <typename T>
struct select_device_functor<T, detail::hip_device_allocator<T>> {
  void operator()(const size_t device_id) { hipSetDevice(device_id); }
};
} // namespace device_selection

/// Recycling allocator for HIP pinned host memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_host =
    detail::aggressive_recycle_allocator<T, detail::hip_pinned_allocator<T>>;
/// Recycling allocator for HIP device memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_device =
    detail::recycle_allocator<T, detail::hip_device_allocator<T>>;

} // namespace memory_recycling
} // end namespace cppuddle
#endif
