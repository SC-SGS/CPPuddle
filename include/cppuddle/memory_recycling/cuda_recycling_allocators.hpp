// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_RECYCLING_ALLOCATORS_HPP
#define CUDA_RECYCLING_ALLOCATORS_HPP

#include "buffer_management_interface.hpp"
// import cuda_pinned_allocator and cuda_device_allocator
#include "detail/cuda_underlying_allocators.hpp"

namespace cppuddle {
namespace memory_recycling {

// Tell cppuddle how to select the device for the cuda allocators
namespace device_selection {
/// GPU device selector using the CUDA API for pinned host allocations
template <typename T>
struct select_device_functor<T, detail::cuda_pinned_allocator<T>> {
  void operator()(const size_t device_id) { cudaSetDevice(device_id); }
};
/// GPU selector using the CUDA API for pinned host allocations
template <typename T>
struct select_device_functor<T, detail::cuda_device_allocator<T>> {
  void operator()(const size_t device_id) { cudaSetDevice(device_id); }
};
} // namespace device_selection

/// Recycling allocator for CUDA pinned host memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_host =
    detail::aggressive_recycle_allocator<T, detail::cuda_pinned_allocator<T>>;
/// Recycling allocator for CUDA device memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_device =
    detail::recycle_allocator<T, detail::cuda_device_allocator<T>>;

/// RAII wrapper for CUDA device memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct cuda_device_buffer {
  recycle_allocator_cuda_device<T> allocator;
  T *device_side_buffer;
  size_t number_of_elements;

  cuda_device_buffer(const size_t number_of_elements, const size_t device_id = 0)
      : allocator{device_id}, number_of_elements(number_of_elements) {
    assert(device_id < max_number_gpus);
    device_side_buffer =
        allocator.allocate(number_of_elements);
  }
  ~cuda_device_buffer() {
    allocator.deallocate(device_side_buffer, number_of_elements);
  }
  // not yet implemented
  cuda_device_buffer(cuda_device_buffer const &other) = delete;
  cuda_device_buffer operator=(cuda_device_buffer const &other) = delete;
  cuda_device_buffer(cuda_device_buffer const &&other) = delete;
  cuda_device_buffer operator=(cuda_device_buffer const &&other) = delete;

};

/// RAII wrapper for CUDA device memory using a passed aggregated allocator
template <typename T, typename Host_Allocator, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct cuda_aggregated_device_buffer {
  T *device_side_buffer;
  size_t number_of_elements;
  cuda_aggregated_device_buffer(size_t number_of_elements, Host_Allocator &alloc)
      : number_of_elements(number_of_elements), alloc(alloc) {
    device_side_buffer =
        alloc.allocate(number_of_elements);
  }
  ~cuda_aggregated_device_buffer() {
    alloc.deallocate(device_side_buffer, number_of_elements);
  }
  // not yet implemented
  cuda_aggregated_device_buffer(cuda_aggregated_device_buffer const &other) = delete;
  cuda_aggregated_device_buffer operator=(cuda_aggregated_device_buffer const &other) = delete;
  cuda_aggregated_device_buffer(cuda_aggregated_device_buffer const &&other) = delete;
  cuda_aggregated_device_buffer operator=(cuda_aggregated_device_buffer const &&other) = delete;

private:
  Host_Allocator &alloc; // will stay valid for the entire aggregation region and hence
                         // for the entire lifetime of this buffer
};

} // namespace memory_recycling
} // end namespace cppuddle
#endif
