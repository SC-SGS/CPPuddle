// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef KOKKOS_BUFFER_UTIL_HPP
#define KOKKOS_BUFFER_UTIL_HPP
#include <Kokkos_Core.hpp>
#include <memory>

namespace recycler {

template<typename element_type, typename alloc_type>
struct view_deleter {
  alloc_type allocator;
  size_t total_elements;
  view_deleter(alloc_type &alloc, size_t total_elements) : allocator(alloc),
    total_elements(total_elements) {}
  void operator()(element_type* p) {
    allocator.deallocate(p, total_elements);
  }
};

template <typename kokkos_type, typename alloc_type, typename element_type>
class aggregated_recycled_view : public kokkos_type {
private:
  alloc_type allocator;
  size_t total_elements{0};
  std::shared_ptr<element_type> data_ref_counter;

public:
  using view_type = kokkos_type;
  template <class... Args>
  explicit aggregated_recycled_view(alloc_type &alloc, Args... args)
      : kokkos_type(
            alloc.allocate(kokkos_type::required_allocation_size(args...) /
                           sizeof(element_type)),
            args...),
        total_elements(kokkos_type::required_allocation_size(args...) /
                       sizeof(element_type)),
        allocator(alloc),
        data_ref_counter(this->data(), view_deleter<element_type, alloc_type>(
                                           alloc, total_elements)) {}

  aggregated_recycled_view(
      const aggregated_recycled_view<kokkos_type, alloc_type, element_type> &other)
      : kokkos_type(other), allocator(other.allocator) {
    data_ref_counter = other.data_ref_counter;
    total_elements = other.total_elements;
  }

  aggregated_recycled_view<kokkos_type, alloc_type, element_type> &
  operator=(const aggregated_recycled_view<kokkos_type, alloc_type, element_type> &other) {
    data_ref_counter = other.data_ref_counter;
    allocator = other.allocator;
    kokkos_type::operator=(other);
    total_elements = other.total_elements;
    return *this;
  }

  aggregated_recycled_view(
      aggregated_recycled_view<kokkos_type, alloc_type, element_type> &&other) noexcept
      : kokkos_type(other), allocator(other.allocator) {
    data_ref_counter = other.data_ref_counter;
    total_elements = other.total_elements;
  }

  aggregated_recycled_view<kokkos_type, alloc_type, element_type> &operator=(
      aggregated_recycled_view<kokkos_type, alloc_type, element_type> &&other) noexcept {
    data_ref_counter = other.data_ref_counter;
    allocator = other.allocator;
    kokkos_type::operator=(other);
    total_elements = other.total_elements;
    return *this;
  }

  ~aggregated_recycled_view() {}
};

template <typename kokkos_type, typename alloc_type, typename element_type>
class recycled_view : public kokkos_type {
private:
  alloc_type allocator;
  size_t total_elements{0};
  std::shared_ptr<element_type> data_ref_counter;

public:
  using view_type = kokkos_type;
  template <class... Args>
  explicit recycled_view(Args... args)
      : kokkos_type(
            allocator.allocate(kokkos_type::required_allocation_size(args...) /
                               sizeof(element_type)),
            args...),
        total_elements(kokkos_type::required_allocation_size(args...) /
                       sizeof(element_type)),
        data_ref_counter(this->data(), view_deleter<element_type, alloc_type>(
                                           allocator, total_elements)) {}

  template <class... Args>
  explicit recycled_view(std::size_t location_id, alloc_type alloc, Args... args)
      : allocator(alloc), kokkos_type(
            allocator.allocate(kokkos_type::required_allocation_size(args...) /
                               sizeof(element_type), location_id),
            args...),
        total_elements(kokkos_type::required_allocation_size(args...) /
                       sizeof(element_type)),
        data_ref_counter(this->data(), view_deleter<element_type, alloc_type>(
                                           allocator, total_elements)) {}

  template <bool use_custom_location_id, class... Args>
  explicit recycled_view(std::size_t location_id, alloc_type alloc, Args... args)
      : allocator(alloc), kokkos_type(
            allocator.allocate(kokkos_type::required_allocation_size(args...) /
                               sizeof(element_type), location_id),
            args...),
        total_elements(kokkos_type::required_allocation_size(args...) /
                       sizeof(element_type)),
        data_ref_counter(this->data(), view_deleter<element_type, alloc_type>(
                                           allocator, total_elements)) {}

  recycled_view(
      const recycled_view<kokkos_type, alloc_type, element_type> &other)
      : kokkos_type(other) {
    total_elements = other.total_elements;
    data_ref_counter = other.data_ref_counter;

  }

  recycled_view<kokkos_type, alloc_type, element_type> &
  operator=(const recycled_view<kokkos_type, alloc_type, element_type> &other) {
    data_ref_counter = other.data_ref_counter;
    kokkos_type::operator=(other);
    total_elements = other.total_elements;
    return *this;
  }

  recycled_view(
      recycled_view<kokkos_type, alloc_type, element_type> &&other) noexcept
      : kokkos_type(other) {
    data_ref_counter = other.data_ref_counter;
    total_elements = other.total_elements;
  }

  recycled_view<kokkos_type, alloc_type, element_type> &operator=(
      recycled_view<kokkos_type, alloc_type, element_type> &&other) noexcept {
    data_ref_counter = other.data_ref_counter;
    kokkos_type::operator=(other);
    total_elements = other.total_elements;
    return *this;
  }

  ~recycled_view() {  }
};

/* template <class kokkos_type, class alloc_type, class element_type> */
/* alloc_type recycled_view<kokkos_type, alloc_type, element_type>::allocator; */

} // end namespace recycler

#endif
