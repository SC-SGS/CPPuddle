// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HIP_RECYCLING_ALLOCATORS_HPP
#define HIP_RECYCLING_ALLOCATORS_HPP

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

#include "buffer_management_interface.hpp"

namespace cppuddle {
namespace memory_recycling {

namespace detail {
/// Underlying host allocator for HIP pinned memory
template <class T> struct hip_pinned_allocator {
  using value_type = T;
  hip_pinned_allocator() noexcept = default;
  template <class U>
  explicit hip_pinned_allocator(hip_pinned_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data;
    // hipError_t error =
    //     hipMallocHost(reinterpret_cast<void **>(&data), n * sizeof(T));
    
    // Even though marked as deprecated, the HIP docs recommend using hipHostMalloc 
    // (not hipMallocHost) for async memcpys 
    // https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Memory-Management.html#hipmemcpyasync
    hipError_t error =
        hipHostMalloc(reinterpret_cast<void **>(&data), n * sizeof(T));
    if (error != hipSuccess) {
      std::string msg =
          std::string(
              "hip_pinned_allocator failed due to hipMallocHost failure : ") +
          std::string(hipGetErrorString(error));
      throw std::runtime_error(msg);
    }
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    hipError_t error = hipHostFree(p);
    if (error != hipSuccess) {
      std::string msg =
          std::string(
              "hip_pinned_allocator failed due to hipFreeHost failure : ") +
          std::string(hipGetErrorString(error));
      throw std::runtime_error(msg);
    }
  }
};
template <class T, class U>
constexpr bool operator==(hip_pinned_allocator<T> const &,
                          hip_pinned_allocator<U> const &) noexcept {
  return true;
}
template <class T, class U>
constexpr bool operator!=(hip_pinned_allocator<T> const &,
                          hip_pinned_allocator<U> const &) noexcept {
  return false;
}

/// Underlying allocator for HIP device memory
template <class T> struct hip_device_allocator {
  using value_type = T;
  hip_device_allocator() noexcept = default;
  template <class U>
  explicit hip_device_allocator(hip_device_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data;
    hipError_t error = hipMalloc(&data, n * sizeof(T));
    if (error != hipSuccess) {
      std::string msg =
          std::string(
              "hip_device_allocator failed due to hipMalloc failure : ") +
          std::string(hipGetErrorString(error));
      throw std::runtime_error(msg);
    }
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    hipError_t error = hipFree(p);
    if (error != hipSuccess) {
      std::string msg =
          std::string(
              "hip_device_allocator failed due to hipFree failure : ") +
          std::string(hipGetErrorString(error));
      throw std::runtime_error(msg);
    }
  }
};
template <class T, class U>
constexpr bool operator==(hip_device_allocator<T> const &,
                          hip_device_allocator<U> const &) noexcept {
  return true;
}
template <class T, class U>
constexpr bool operator!=(hip_device_allocator<T> const &,
                          hip_device_allocator<U> const &) noexcept {
  return false;
}

} // end namespace detail


namespace device_selection {
/// GPU device selector using the HIP API for pinned host allocations
template <typename T>
struct select_device_functor<T, detail::hip_pinned_allocator<T>> {
  void operator()(const size_t device_id) { hipSetDevice(device_id); }
};
/// GPU selector using the HIP API for pinned host allocations
template <typename T>
struct select_device_functor<T, detail::hip_device_allocator<T>> {
  void operator()(const size_t device_id) { hipSetDevice(device_id); }
};
} // namespace device_selection

/// Recycling allocator for HIP pinned host memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_host =
    detail::aggressive_recycle_allocator<T, detail::hip_pinned_allocator<T>>;
/// Recycling allocator for HIP device memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_device =
    detail::recycle_allocator<T, detail::hip_device_allocator<T>>;

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
