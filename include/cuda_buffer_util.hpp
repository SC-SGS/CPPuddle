// Copyright (c) 2020-2023 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_BUFFER_UTIL_HPP
#define CUDA_BUFFER_UTIL_HPP

#include "buffer_manager.hpp"
#include "detail/config.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace recycler {

namespace detail {

template <class T> struct cuda_pinned_allocator {
  using value_type = T;
  cuda_pinned_allocator() noexcept = default;
  template <class U>
  explicit cuda_pinned_allocator(cuda_pinned_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    cudaSetDevice(get_device_id());
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

template <class T> struct cuda_device_allocator {
  using value_type = T;
  cuda_device_allocator() noexcept = default;
  template <class U>
  explicit cuda_device_allocator(cuda_device_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    cudaSetDevice(get_device_id());
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

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_host =
    detail::aggressive_recycle_allocator<T, detail::cuda_pinned_allocator<T>>;
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_cuda_device =
    detail::recycle_allocator<T, detail::cuda_device_allocator<T>>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct cuda_device_buffer {
  size_t gpu_id{0};
  T *device_side_buffer;
  size_t number_of_elements;
  explicit cuda_device_buffer(size_t number_of_elements)
      : number_of_elements(number_of_elements) {
    device_side_buffer =
        recycle_allocator_cuda_device<T>{}.allocate(number_of_elements);
  }
  // TODO deprecate and remove gpu_id
  explicit cuda_device_buffer(size_t number_of_elements, size_t gpu_id)
      : gpu_id(gpu_id), number_of_elements(number_of_elements), set_id(true) {
    assert(gpu_id == 0);
    device_side_buffer =
        recycle_allocator_cuda_device<T>{}.allocate(number_of_elements);
  }
  ~cuda_device_buffer() {
    recycle_allocator_cuda_device<T>{}.deallocate(device_side_buffer,
                                                  number_of_elements);
  }
  // not yet implemented
  cuda_device_buffer(cuda_device_buffer const &other) = delete;
  cuda_device_buffer operator=(cuda_device_buffer const &other) = delete;
  cuda_device_buffer(cuda_device_buffer const &&other) = delete;
  cuda_device_buffer operator=(cuda_device_buffer const &&other) = delete;

private:
  bool set_id{false};
};

template <typename T, typename Host_Allocator, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct cuda_aggregated_device_buffer {
  size_t gpu_id{0};
  T *device_side_buffer;
  size_t number_of_elements;
  explicit cuda_aggregated_device_buffer(size_t number_of_elements)
      : number_of_elements(number_of_elements) {
    device_side_buffer =
        recycle_allocator_cuda_device<T>{}.allocate(number_of_elements);
  }
  // TODO deprecate and remove gpu_id
  explicit cuda_aggregated_device_buffer(size_t number_of_elements, size_t gpu_id, Host_Allocator &alloc)
      : gpu_id(gpu_id), number_of_elements(number_of_elements), set_id(true), alloc(alloc) {
    assert(gpu_id == 0);
    device_side_buffer =
        alloc.allocate(number_of_elements);
  }
  ~cuda_aggregated_device_buffer() {
    alloc.deallocate(device_side_buffer,
                                                  number_of_elements);
  }
  // not yet implemented
  cuda_aggregated_device_buffer(cuda_aggregated_device_buffer const &other) = delete;
  cuda_aggregated_device_buffer operator=(cuda_aggregated_device_buffer const &other) = delete;
  cuda_aggregated_device_buffer(cuda_aggregated_device_buffer const &&other) = delete;
  cuda_aggregated_device_buffer operator=(cuda_aggregated_device_buffer const &&other) = delete;

private:
  bool set_id{false};
  Host_Allocator &alloc;
};

} // end namespace recycler
#endif
