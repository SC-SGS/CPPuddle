// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_RECYCLING_ALLOCATORS_HPP
#define SYCL_RECYCLING_ALLOCATORS_HPP

#include "buffer_management_interface.hpp"
#include "detail/sycl_underlying_allocators.hpp"

namespace cppuddle {
namespace memory_recycling {

namespace device_selection {
// No MutliGPU support yet, hence no select_device_function required
static_assert(max_number_gpus <= 1, "CPPuddle currently does not support MultiGPU SYCL builds!");
} // namespace device_selection

/// Recycling allocator for SYCL pinned host memory (default device)
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_sycl_host =
    detail::aggressive_recycle_allocator<T, detail::sycl_host_default_allocator<T>>;
/// Recycling allocator for SYCL device memory (default device)
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_sycl_device =
    detail::recycle_allocator<T, detail::sycl_device_default_allocator<T>>;

} // namespace memory_recycling
} // end namespace cppuddle
#endif
