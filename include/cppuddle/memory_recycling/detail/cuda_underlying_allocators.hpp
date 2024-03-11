// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_UNDERLYING_ALLOCATORS_HPP
#define CUDA_UNDERLYING_ALLOCATORS_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cppuddle {
namespace memory_recycling {
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
} // namespace memory_recycling
} // end namespace cppuddle

#endif
