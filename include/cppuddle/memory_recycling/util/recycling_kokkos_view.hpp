// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef KOKKOS_BUFFER_UTIL_HPP
#define KOKKOS_BUFFER_UTIL_HPP
#include <Kokkos_Core.hpp>
#include <memory>
#include <type_traits>

#include "cppuddle/memory_recycling/buffer_management_interface.hpp"


namespace cppuddle {
namespace memory_recycling {


template<typename element_type, typename alloc_type>
struct view_deleter {
  alloc_type allocator;
  size_t total_elements;
  view_deleter(alloc_type alloc, size_t total_elements) : allocator(alloc),
    total_elements(total_elements) {}
  void operator()(element_type* p) {
    allocator.deallocate(p, total_elements);
  }
};

template <typename kokkos_type, typename alloc_type, typename element_type>
class aggregated_recycling_view : public kokkos_type {
private:
  alloc_type allocator;
  size_t total_elements{0};
  std::shared_ptr<element_type> data_ref_counter;
  static_assert(std::is_same_v<element_type, typename alloc_type::value_type>);

public:
  using view_type = kokkos_type;
  template <class... Args>
  explicit aggregated_recycling_view(alloc_type &alloc, Args... args)
      : kokkos_type(
            alloc.allocate(kokkos_type::required_allocation_size(args...) /
                           sizeof(element_type)),
            args...),
        total_elements(kokkos_type::required_allocation_size(args...) /
                       sizeof(element_type)),
        allocator(alloc),
        data_ref_counter(this->data(), view_deleter<element_type, alloc_type>(
                                           alloc, total_elements)) {}

  aggregated_recycling_view(
      const aggregated_recycling_view<kokkos_type, alloc_type, element_type> &other)
      : kokkos_type(other), allocator(other.allocator) {
    data_ref_counter = other.data_ref_counter;
    total_elements = other.total_elements;
  }

  aggregated_recycling_view<kokkos_type, alloc_type, element_type> &
  operator=(const aggregated_recycling_view<kokkos_type, alloc_type, element_type> &other) {
    data_ref_counter = other.data_ref_counter;
    allocator = other.allocator;
    kokkos_type::operator=(other);
    total_elements = other.total_elements;
    return *this;
  }

  aggregated_recycling_view(
      aggregated_recycling_view<kokkos_type, alloc_type, element_type> &&other) noexcept
      : kokkos_type(other), allocator(other.allocator) {
    data_ref_counter = other.data_ref_counter;
    total_elements = other.total_elements;
  }

  aggregated_recycling_view<kokkos_type, alloc_type, element_type> &operator=(
      aggregated_recycling_view<kokkos_type, alloc_type, element_type> &&other) noexcept {
    data_ref_counter = other.data_ref_counter;
    allocator = other.allocator;
    kokkos_type::operator=(other);
    total_elements = other.total_elements;
    return *this;
  }

  ~aggregated_recycling_view() {}
};


template <typename kokkos_type, typename alloc_type, typename element_type>
class recycling_view : public kokkos_type {
private:
  size_t total_elements{0};
  std::shared_ptr<element_type> data_ref_counter;

public:
  using view_type = kokkos_type;
  static_assert(std::is_same_v<element_type, typename alloc_type::value_type>);
  template <typename... Args,
            std::enable_if_t<sizeof...(Args) == kokkos_type::rank, bool> = true>
  recycling_view(Args... args)
      : kokkos_type(
            alloc_type{}.allocate(kokkos_type::required_allocation_size(args...) /
                               sizeof(element_type)),
            args...),
        total_elements(kokkos_type::required_allocation_size(args...) /
                       sizeof(element_type)),
        data_ref_counter(this->data(), view_deleter<element_type, alloc_type>(
                                           alloc_type{}, total_elements)) {}

  template <typename... Args,
            std::enable_if_t<sizeof...(Args) == kokkos_type::rank, bool> = true>
  recycling_view(const size_t device_id, Args... args)
      : kokkos_type(
            alloc_type{device_id}.allocate(kokkos_type::required_allocation_size(args...) /
                               sizeof(element_type)),
            args...),
        total_elements(kokkos_type::required_allocation_size(args...) /
                       sizeof(element_type)),
        data_ref_counter(this->data(), view_deleter<element_type, alloc_type>(
                                           alloc_type{device_id}, total_elements)) {}

  template <
      typename layout_t,
      std::enable_if_t<Kokkos::is_array_layout<layout_t>::value, bool> = true>
  recycling_view(std::size_t device_id, layout_t layout)
      : kokkos_type(
            alloc_type{device_id}.allocate(kokkos_type::required_allocation_size(layout) /
                               sizeof(element_type)),
            layout),
        total_elements(kokkos_type::required_allocation_size(layout) /
                       sizeof(element_type)),
        data_ref_counter(this->data(), view_deleter<element_type, alloc_type>(
                                           alloc_type{device_id}, total_elements)) {}

  recycling_view(
      const recycling_view<kokkos_type, alloc_type, element_type> &other)
      : kokkos_type(other) {
    total_elements = other.total_elements;
    data_ref_counter = other.data_ref_counter;

  }

  recycling_view<kokkos_type, alloc_type, element_type> &
  operator=(const recycling_view<kokkos_type, alloc_type, element_type> &other) {
    data_ref_counter = other.data_ref_counter;
    kokkos_type::operator=(other);
    total_elements = other.total_elements;
    return *this;
  }

  recycling_view(
      recycling_view<kokkos_type, alloc_type, element_type> &&other) noexcept
      : kokkos_type(other) {
    data_ref_counter = other.data_ref_counter;
    total_elements = other.total_elements;
  }

  recycling_view<kokkos_type, alloc_type, element_type> &operator=(
      recycling_view<kokkos_type, alloc_type, element_type> &&other) noexcept {
    data_ref_counter = other.data_ref_counter;
    kokkos_type::operator=(other);
    total_elements = other.total_elements;
    return *this;
  }

  ~recycling_view() {  }
};

} // namespace memory_recycling
} // end namespace cppuddle

#endif
