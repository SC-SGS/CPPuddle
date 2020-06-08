#ifndef CUDA_BUFFER_UTIL_HPP
#define CUDA_BUFFER_UTIL_HPP

#include "buffer_manager.hpp"

#include <cuda_runtime.h>

namespace recycler {

namespace detail {

template <class T> struct cuda_pinned_allocator {
  using value_type = T;
  cuda_pinned_allocator() noexcept = default;
  template <class U>
  explicit cuda_pinned_allocator(cuda_pinned_allocator<U> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data;
    cudaMallocHost(reinterpret_cast<void **>(&data), n * sizeof(T));
    return data;
  }
  void deallocate(T *p, std::size_t n) { cudaFreeHost(p); }
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
    T *data;
    cudaMalloc(&data, n * sizeof(T));
    return data;
  }
  void deallocate(T *p, std::size_t n) { cudaFree(p); }
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
  T *device_side_buffer;
  size_t number_of_elements;
  explicit cuda_device_buffer(size_t number_of_elements)
      : number_of_elements(number_of_elements) {
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
};

} // end namespace recycler
#endif
