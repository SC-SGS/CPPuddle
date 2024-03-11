// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_RECYCLING_BUFFER_HPP
#define CUDA_RECYCLING_BUFFER_HPP

// import recycle_allocator_cuda_device
#include "cppuddle/memory_recycling/cuda_recycling_allocators.hpp""

/// \file
/// Contains a RAII wrappers for CUDA device buffers. Intended to be used with
/// the recycling allocators but technically any allocator should work

namespace cppuddle {
namespace memory_recycling {


/// RAII wrapper for CUDA device memory
/// (ideally used with a recycling allocator)
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
/// (which ideally should be an allocator_slice from the work aggregation)
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
