// Copyright (c: 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_RECYCLING_ALLOCATORS_HPP
#define SYCL_RECYCLING_ALLOCATORS_HPP

#include <CL/sycl.hpp>
#include <stdexcept>
#include <string>

#include "buffer_management_interface.hpp"

namespace cppuddle {
namespace memory_recycling {

namespace device_selection {
// No MutliGPU support yet, hence no select_device_function required
static_assert(max_number_gpus == 1, "CPPuddle currently does not support MultiGPU SYCL builds!");
} // namespace device_selection

namespace detail {
/// Underlying host allocator for SYCL pinned memory (using the sycl::default_selector{})
template <class T> struct sycl_host_default_allocator {
  using value_type = T;
  sycl_host_default_allocator() noexcept = default;
  template <class U>
  explicit sycl_host_default_allocator(sycl_host_default_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    static cl::sycl::queue default_queue(cl::sycl::default_selector{});
    T *data = cl::sycl::malloc_host<T>(n, default_queue);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    static cl::sycl::queue default_queue(cl::sycl::default_selector{});
    cl::sycl::free(p, default_queue);
  }
};
template <class T, class U>
constexpr bool operator==(sycl_host_default_allocator<T> const &,
                          sycl_host_default_allocator<U> const &) noexcept {
  return true;
}
template <class T, class U>
constexpr bool operator!=(sycl_host_default_allocator<T> const &,
                          sycl_host_default_allocator<U> const &) noexcept {
  return false;
}

/// Underlying allocator for SYCL device memory (using the sycl::default_selector{})
template <class T> struct sycl_device_default_allocator {
  using value_type = T;
  sycl_device_default_allocator() noexcept = default;
  template <class U>
  explicit sycl_device_default_allocator(sycl_device_default_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    static cl::sycl::queue default_queue(cl::sycl::default_selector{});
    T *data = cl::sycl::malloc_device<T>(n, default_queue);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    static cl::sycl::queue default_queue(cl::sycl::default_selector{});
    cl::sycl::free(p, default_queue);
  }
};
template <class T, class U>
constexpr bool operator==(sycl_device_default_allocator<T> const &,
                          sycl_device_default_allocator<U> const &) noexcept {
  return true;
}
template <class T, class U>
constexpr bool operator!=(sycl_device_default_allocator<T> const &,
                          sycl_device_default_allocator<U> const &) noexcept {
  return false;
}

} // end namespace detail

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
