// Copyright (c: 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_BUFFER_UTIL_HPP
#define CUDA_BUFFER_UTIL_HPP

#include "buffer_manager.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace recycler {

namespace detail {

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

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_host =
    detail::aggressive_recycle_allocator<T, detail::hip_pinned_allocator<T>>;
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_allocator_hip_device =
    detail::recycle_allocator<T, detail::hip_device_allocator<T>>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct hip_device_buffer {
  size_t gpu_id{0};
  T *device_side_buffer;
  size_t number_of_elements;
  explicit hip_device_buffer(size_t number_of_elements)
      : number_of_elements(number_of_elements) {
    device_side_buffer =
        recycle_allocator_hip_device<T>{}.allocate(number_of_elements);
  }
  explicit hip_device_buffer(size_t number_of_elements, size_t gpu_id)
      : gpu_id(gpu_id), number_of_elements(number_of_elements), set_id(true) {

    // TODO Fix Multi GPU support
    // hipSetDevice(gpu_id);
    device_side_buffer =
        recycle_allocator_hip_device<T>{}.allocate(number_of_elements);
  }
  ~hip_device_buffer() {
    // TODO Fix Multi GPU support
    // if (set_id)
    //   hipSetDevice(gpu_id);
    recycle_allocator_hip_device<T>{}.deallocate(device_side_buffer,
                                                  number_of_elements);
  }
  // not yet implemented
  hip_device_buffer(hip_device_buffer const &other) = delete;
  hip_device_buffer operator=(hip_device_buffer const &other) = delete;
  hip_device_buffer(hip_device_buffer const &&other) = delete;
  hip_device_buffer operator=(hip_device_buffer const &&other) = delete;

private:
  bool set_id{false};
};

} // end namespace recycler
#endif
