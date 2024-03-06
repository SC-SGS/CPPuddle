// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_RECYCLING_ALLOCATORS_HPP
#define CUDA_RECYCLING_ALLOCATORS_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "buffer_management_interface.hpp"

namespace cppuddle {
namespace detail {

/// Underlying host allocator for CUDA pinned memory
template <class T> struct cuda_pinned_allocator {
  using value_type = T;
  cuda_pinned_allocator() noexcept = default;
  template <class U>
  explicit cuda_pinned_allocator(cuda_pinned_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data;
    cudaError_t error =
        cudaMallocHost(reinterpret_cast<void **>(&data), n * sizeof(T));
    if (error != cudaSuccess) {
      std::string msg =
          std::string(
              "cuda_pinned_allocator failed due to cudaMallocHost failure : ") +
          std::string(cudaGetErrorString(error));
      throw std::runtime_error(msg);
    }
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    cudaError_t error = cudaFreeHost(p);
    if (error != cudaSuccess) {
      std::string msg =
          std::string(
              "cuda_pinned_allocator failed due to cudaFreeHost failure : ") +
          std::string(cudaGetErrorString(error));
      throw std::runtime_error(msg);
    }
  }
};

template <class T, class U>
constexpr bool operator==(cuda_pinned_allocator<T> const &,
                          cuda_pinned_allocator<U> const &) noexcept {
  return true;
}
template <class T, class U>
constexpr bool operator!=(cuda_pinned_allocator<T> const &,
                          cuda_pinned_allocator<U> const &) noexcept {
  return false;
}

/// Underlying allocator for CUDA device memory
template <class T> struct cuda_device_allocator {
  using value_type = T;
  cuda_device_allocator() noexcept = default;
  template <class U>
  explicit cuda_device_allocator(cuda_device_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data;
    cudaError_t error = cudaMalloc(&data, n * sizeof(T));
    if (error != cudaSuccess) {
      std::string msg =
          std::string(
              "cuda_device_allocator failed due to cudaMalloc failure : ") +
          std::string(cudaGetErrorString(error));
      throw std::runtime_error(msg);
    }
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    cudaError_t error = cudaFree(p);
    if (error != cudaSuccess) {
      std::string msg =
          std::string(
              "cuda_device_allocator failed due to cudaFree failure : ") +
          std::string(cudaGetErrorString(error));
      throw std::runtime_error(msg);
    }
  }
};
template <class T, class U>
constexpr bool operator==(cuda_device_allocator<T> const &,
                          cuda_device_allocator<U> const &) noexcept {
  return true;
}
template <class T, class U>
constexpr bool operator!=(cuda_device_allocator<T> const &,
                          cuda_device_allocator<U> const &) noexcept {
  return false;
}
} // end namespace detail


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
} // end namespace cppuddle
#endif
