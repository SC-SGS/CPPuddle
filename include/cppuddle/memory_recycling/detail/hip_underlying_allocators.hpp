// Copyright (c) 2021-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HIP_UNDERLYING_ALLOCATORS_HPP
#define HIP_UNDERLYING_ALLOCATORS_HPP

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

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
} // namespace memory_recycling
} // end namespace cppuddle

#endif
