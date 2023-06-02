// Copyright (c) 2023 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#ifndef CPPUDDLE_HPX_BUFFER_UTIL_HPP
#define CPPUDDLE_HPX_BUFFER_UTIL_HPP

#include "buffer_manager.hpp"
#include <hpx/include/runtime.hpp>

namespace recycler {
namespace detail {

template <typename T, typename Host_Allocator> struct numa_aware_recycle_allocator {
  using value_type = T;
  numa_aware_recycle_allocator() noexcept = default;
  size_t dealloc_hint{0};
  template <typename U>
  explicit numa_aware_recycle_allocator(
      numa_aware_recycle_allocator<U, Host_Allocator> const &) noexcept {}
  T *allocate(std::size_t n) {
    dealloc_hint = hpx::get_worker_thread_num();
    T *data = buffer_recycler::get<T, Host_Allocator>(n, false, dealloc_hint);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n, dealloc_hint);
  }
  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    ::new (static_cast<void *>(p)) T(std::forward<Args>(args)...);
  }
  void destroy(T *p) { p->~T(); }
};
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator==(numa_aware_recycle_allocator<T, Host_Allocator> const &,
           numa_aware_recycle_allocator<U, Host_Allocator> const &) noexcept {
  return true;
}
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator!=(numa_aware_recycle_allocator<T, Host_Allocator> const &,
           numa_aware_recycle_allocator<U, Host_Allocator> const &) noexcept {
  return false;
}

/// Recycles not only allocations but also the contents of a buffer
template <typename T, typename Host_Allocator>
struct numa_aware_aggressive_recycle_allocator {
  using value_type = T;
  numa_aware_aggressive_recycle_allocator() noexcept = default;
  size_t dealloc_hint{0};
  template <typename U>
  explicit numa_aware_aggressive_recycle_allocator(
      numa_aware_aggressive_recycle_allocator<U, Host_Allocator> const &) noexcept {}
  T *allocate(std::size_t n) {
    dealloc_hint = hpx::get_worker_thread_num();
    T *data = buffer_recycler::get<T, Host_Allocator>(
        n, true, dealloc_hint); // also initializes the buffer if it isn't reused
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n, dealloc_hint);
  }
  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    // Do nothing here - we reuse the content of the last owner
  }
  void destroy(T *p) {
    // Do nothing here - Contents will be destroyed when the buffer manager is
    // destroyed, not before
  }
};
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator==(numa_aware_aggressive_recycle_allocator<T, Host_Allocator> const &,
           numa_aware_aggressive_recycle_allocator<U, Host_Allocator> const &) noexcept {
  return true;
}
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator!=(numa_aware_aggressive_recycle_allocator<T, Host_Allocator> const &,
           numa_aware_aggressive_recycle_allocator<U, Host_Allocator> const &) noexcept {
  return false;
}

}
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using numa_aware_recycle_std = detail::numa_aware_recycle_allocator<T, std::allocator<T>>;
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using numa_aware_aggressive_recycle_std =
    detail::numa_aware_aggressive_recycle_allocator<T, std::allocator<T>>;
}

#endif
