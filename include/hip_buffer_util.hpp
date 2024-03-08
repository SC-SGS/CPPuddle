// Copyright (c): 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HIP_BUFFER_UTIL_HPP
#define HIP_BUFFER_UTIL_HPP

#include "/cppuddle/memory_recycling/hip_recycling_allocators.hpp"

namespace recycler {

namespace detail {

template <class T>
using hip_pinned_allocator
    [[deprecated("Use from header hip_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::detail::hip_pinned_allocator<T>;

template <class T>
using hip_device_allocator
    [[deprecated("Use from header hip_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::detail::hip_device_allocator<T>;
} // end namespace detail

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_host
    [[deprecated("Use from header hip_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::recycle_allocator_hip_host<T>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_device
    [[deprecated("Use from header hip_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::recycle_allocator_hip_device<T>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using hip_device_buffer
    [[deprecated("Use from header hip_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::hip_device_buffer<T>;

template <typename T, typename Host_Allocator,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using hip_aggregated_device_buffer
    [[deprecated("Use from header hip_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::hip_aggregated_device_buffer<T, Host_Allocator>;

} // end namespace recycler
#endif
