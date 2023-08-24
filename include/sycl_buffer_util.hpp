// Copyright (c: 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_BUFFER_UTIL_HPP
#define SYCL_BUFFER_UTIL_HPP

#include "buffer_manager.hpp"

#include <CL/sycl.hpp>
#include <stdexcept>
#include <string>

namespace recycler {

namespace detail {

static_assert(max_number_gpus == 1, "CPPuddle currently does not support MultiGPU SYCL builds!");

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

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_sycl_host =
    detail::aggressive_recycle_allocator<T, detail::sycl_host_default_allocator<T>>;
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_sycl_device =
    detail::recycle_allocator<T, detail::sycl_device_default_allocator<T>>;

} // end namespace recycler
#endif
