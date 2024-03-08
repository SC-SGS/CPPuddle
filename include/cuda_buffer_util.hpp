// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_BUFFER_UTIL_HPP
#define CUDA_BUFFER_UTIL_HPP

#include "buffer_manager.hpp"
#include "cppuddle/memory_recycling/cuda_recycling_allocators.hpp"

namespace recycler {
namespace detail {

template <class T>
using cuda_pinned_allocator
    [[deprecated("Use from header cuda_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::detail::cuda_pinned_allocator<T>;

template <class T>
using cuda_device_allocator
    [[deprecated("Use from header cuda_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::detail::cuda_device_allocator<T>;

} // end namespace detail

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_host
    [[deprecated("Use from header cuda_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::recycle_allocator_cuda_host<T>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_device
    [[deprecated("Use from header cuda_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::recycle_allocator_cuda_device<T>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using cuda_device_buffer
    [[deprecated("Use from header cuda_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::cuda_device_buffer<T>;

template <typename T, typename Host_Allocator,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using cuda_aggregated_device_buffer
    [[deprecated("Use from header cuda_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::cuda_aggregated_device_buffer<T, Host_Allocator>;

} // end namespace recycler
#endif
