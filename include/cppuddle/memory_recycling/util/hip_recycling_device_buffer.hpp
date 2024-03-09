// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HIP_RECYCLING_BUFFER_HPP
#define HIP_RECYCLING_BUFFER_HPP

// import recycle_allocator_hip_device
#include "cppuddle/memory_recycling/hip_recycling_allocators.hpp"

namespace cppuddle {
namespace memory_recycling {

/// RAII wrapper for HIP device memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct hip_device_buffer {
  recycle_allocator_hip_device<T> allocator;
  T *device_side_buffer;
  size_t number_of_elements;

  hip_device_buffer(size_t number_of_elements, size_t device_id)
      : allocator{device_id}, number_of_elements(number_of_elements) {
    assert(device_id < max_number_gpus);
    device_side_buffer =
        allocator.allocate(number_of_elements);
  }
  ~hip_device_buffer() {
    allocator.deallocate(device_side_buffer, number_of_elements);
  }
  // not yet implemented
  hip_device_buffer(hip_device_buffer const &other) = delete;
  hip_device_buffer operator=(hip_device_buffer const &other) = delete;
  hip_device_buffer(hip_device_buffer const &&other) = delete;
  hip_device_buffer operator=(hip_device_buffer const &&other) = delete;

};

/// RAII wrapper for CUDA device memory using a passed aggregated allocator
template <typename T, typename Host_Allocator, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct hip_aggregated_device_buffer {
  T *device_side_buffer;
  size_t number_of_elements;
  hip_aggregated_device_buffer(size_t number_of_elements, Host_Allocator &alloc)
      : number_of_elements(number_of_elements), alloc(alloc) {
    device_side_buffer =
        alloc.allocate(number_of_elements);
  }
  ~hip_aggregated_device_buffer() {
    alloc.deallocate(device_side_buffer, number_of_elements);
  }
  // not yet implemented
  hip_aggregated_device_buffer(hip_aggregated_device_buffer const &other) = delete;
  hip_aggregated_device_buffer operator=(hip_aggregated_device_buffer const &other) = delete;
  hip_aggregated_device_buffer(hip_aggregated_device_buffer const &&other) = delete;
  hip_aggregated_device_buffer operator=(hip_aggregated_device_buffer const &&other) = delete;

private:
  Host_Allocator &alloc; // will stay valid for the entire aggregation region and hence
                         // for the entire lifetime of this buffer
};

} // namespace memory_recycling
} // end namespace cppuddle
#endif
