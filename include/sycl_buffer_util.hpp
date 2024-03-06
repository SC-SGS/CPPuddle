// Copyright (c: 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_BUFFER_UTIL_HPP
#define SYCL_BUFFER_UTIL_HPP

#include "sycl_recycling_allocators.hpp"

namespace recycler {

namespace detail {

[[deprecated("Use from header sycl_recycling_allocators.hpp instead")]]
template <class T> 
using sycl_host_default_allocator = cppuddle::detail::sycl_host_default_allocator<T>;

[[deprecated("Use from header sycl_recycling_allocators.hpp instead")]]
template <class T> 
using sycl_device_default_allocator = cppuddle::detail::sycl_device_default_allocator<T>;

} // end namespace detail

[[deprecated("Use from header sycl_recycling_allocators.hpp instead")]]
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_sycl_host = cppuddle::recycle_allocator_sycl_host<T>;
[[deprecated("Use from header sycl_recycling_allocators.hpp instead")]]
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_sycl_device = cppuddle::recycle_allocator_sycl_device<T>;

} // end namespace recycler
#endif
