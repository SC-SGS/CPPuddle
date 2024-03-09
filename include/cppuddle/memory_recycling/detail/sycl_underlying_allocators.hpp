// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_UNDERLYING_ALLOCATORS_HPP
#define SYCL_UNDERLYING_ALLOCATORS_HPP

#include <CL/sycl.hpp>
#include <stdexcept>
#include <string>

namespace cppuddle {
namespace memory_recycling {
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
} // namespace memory_recycling
} // end namespace cppuddle

#endif
