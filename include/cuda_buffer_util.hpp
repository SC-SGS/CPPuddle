// Copyright (c) 2020-2023 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_BUFFER_UTIL_HPP
#define CUDA_BUFFER_UTIL_HPP

#include "cuda_recycling_allocators.hpp"
namespace recycler {

namespace detail {

[[deprecated("Use from header cuda_recycling_allocators.hpp instead")]]
template <class T>
using cuda_pinned_allocator = cppuddle::detail::cuda_pinned_allocator<T>;

[[deprecated("Use from header cuda_recycling_allocators.hpp instead")]]
template <class T>
using cuda_device_allocator = cppuddle::detail::cuda_device_allocator<T>;

} // end namespace detail

[[deprecated("Use from header cuda_recycling_allocators.hpp instead")]]
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_host =
    cppuddle::recycle_allocator_cuda_host<T>;
[[deprecated("Use from header cuda_recycling_allocators.hpp instead")]]
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_device =
    cppuddle::recycle_allocator_cuda_device<T>;

[[deprecated("Use from header cuda_recycling_allocators.hpp instead")]]
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using cuda_device_buffer = cppuddle::cuda_device_buffer<T>;

[[deprecated("Use from header cuda_recycling_allocators.hpp instead")]]
template <typename T, typename Host_Allocator, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using cuda_aggregated_device_buffer = cppuddle::cuda_aggregated_device_buffer<T>;

} // end namespace recycler
#endif
