// Copyright (c: 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HIP_BUFFER_UTIL_HPP
#define HIP_BUFFER_UTIL_HPP

#include "hip_recycling_allocators.hpp"

namespace recycler {

namespace detail {

[[deprecated("Use from header hip_recycling_allocators.hpp instead")]]
template <class T> 
using hip_pinned_allocator = cppuddle::detail::hip_pinned_allocator<T>;

[[deprecated("Use from header hip_recycling_allocators.hpp instead")]]
template <class T> 
using hip_device_allocator = cppuddle::detail::hip_device_allocator<T>;
} // end namespace detail

[[deprecated("Use from header hip_recycling_allocators.hpp instead")]]
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_host = cppuddle::recycle_allocator_hip_host<T>;
[[deprecated("Use from header hip_recycling_allocators.hpp instead")]]
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_device = cppuddle::recycle_allocator_hip_device<T>;

[[deprecated("Use from header hip_recycling_allocators.hpp instead")]]
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using hip_device_buffer = cppuddle::hip_device_buffer<T>;

[[deprecated("Use from header hip_recycling_allocators.hpp instead")]]
template <typename T, typename Host_Allocator, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using hip_aggregated_device_buffer = cppuddle::hip_aggregated_device_buffer<T>;

} // end namespace recycler
#endif
