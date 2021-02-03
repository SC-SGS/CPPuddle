// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef STREAM_TEST_HPP // NOLINT
#define STREAM_TEST_HPP // NOLINT
#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"

template <typename Interface, typename Pool, typename... Ts>
void test_pool_memcpy(const size_t stream_parameter, Ts &&... ts) {
  std::vector<double, recycler::recycle_allocator_cuda_host<double>> hostbuffer(
      512);
  recycler::cuda_device_buffer<double> devicebuffer(512);
  stream_pool::init<Interface, Pool>(stream_parameter, std::forward<Ts>(ts)...);
  // without interface wrapper
  {
    auto test1 = stream_pool::get_interface<Interface, Pool>();
    Interface test1_interface = std::get<0>(test1);
    size_t interface_id = std::get<1>(test1);
    test1_interface.post(cudaMemcpyAsync, devicebuffer.device_side_buffer,
                         hostbuffer.data(), 512 * sizeof(double),
                         cudaMemcpyHostToDevice);
    auto fut1 = test1_interface.async_execute(
        cudaMemcpyAsync, hostbuffer.data(), devicebuffer.device_side_buffer,
        512 * sizeof(double), cudaMemcpyDeviceToHost);
    fut1.get();
    stream_pool::release_interface<Interface, Pool>(interface_id);
  }

  // with interface wrapper
  {
    stream_interface<Interface, Pool> test1_interface;
    // hpx::cuda::cuda_executor test1_interface(0, false);
    test1_interface.post(cudaMemcpyAsync, devicebuffer.device_side_buffer,
                         hostbuffer.data(), 512 * sizeof(double),
                         cudaMemcpyHostToDevice);
    auto fut1 = test1_interface.async_execute(
        cudaMemcpyAsync, hostbuffer.data(), devicebuffer.device_side_buffer,
        512 * sizeof(double), cudaMemcpyDeviceToHost);
    fut1.get();
  }
  stream_pool::cleanup<Interface, Pool>();
}

template <typename Interface, typename Pool, typename... Ts>
void test_pool_ref_counting(const size_t stream_parameter, Ts &&... ts) {

  // init ppol
  stream_pool::init<Interface, Pool>(stream_parameter, std::forward<Ts>(ts)...);
  {
    // Allocating
    auto test1 = stream_pool::get_interface<Interface, Pool>();
    auto load1 = stream_pool::get_current_load<Interface, Pool>();
    assert(load1 == 0);
    Interface test1_interface = std::get<0>(test1);
    size_t test1_index = std::get<1>(test1);
    auto test2 = stream_pool::get_interface<Interface, Pool>();
    auto load2 = stream_pool::get_current_load<Interface, Pool>();
    assert(load2 == 1);
    Interface test2_interface = std::get<0>(test2);
    // auto fut = test2_interface.get_future();
    size_t test2_index = std::get<1>(test2);
    auto test3 = stream_pool::get_interface<Interface, Pool>();
    auto load3 = stream_pool::get_current_load<Interface, Pool>();
    assert(load3 == 1);
    Interface test3_interface = std::get<0>(test3);
    size_t test3_index = std::get<1>(test3);
    auto test4 = stream_pool::get_interface<Interface, Pool>();
    auto load4 = stream_pool::get_current_load<Interface, Pool>();
    Interface test4_interface = std::get<0>(test4);
    size_t test4_index = std::get<1>(test4);
    assert(load4 == 2);
    // Releasing
    stream_pool::release_interface<Interface, Pool>(test4_index);
    load4 = stream_pool::get_current_load<Interface, Pool>();
    assert(load4 == 1);
    stream_pool::release_interface<Interface, Pool>(test3_index);
    load3 = stream_pool::get_current_load<Interface, Pool>();
    assert(load3 == 1);
    stream_pool::release_interface<Interface, Pool>(test2_index);
    load2 = stream_pool::get_current_load<Interface, Pool>();
    assert(load2 == 0);
    stream_pool::release_interface<Interface, Pool>(test1_index);
    load1 = stream_pool::get_current_load<Interface, Pool>();
    assert(load1 == 0);
  }
  // Clear
  auto load0 = stream_pool::get_current_load<Interface, Pool>();
  assert(load0 == 0);
  stream_pool::cleanup<Interface, Pool>();
}

template <typename Interface, typename Pool, typename... Ts>
void test_pool_wrappers(const size_t stream_parameter, Ts &&... ts) {
  using wrapper_type = stream_interface<Interface, Pool>;
  // init ppol
  stream_pool::init<Interface, Pool>(stream_parameter, std::forward<Ts>(ts)...);
  {
    wrapper_type test1;
    auto load = stream_pool::get_current_load<Interface, Pool>();
    assert(load == 0);
    wrapper_type test2;
    load = stream_pool::get_current_load<Interface, Pool>();
    // auto fut = test2.get_future();
    assert(load == 1);
    wrapper_type test3;
    load = stream_pool::get_current_load<Interface, Pool>();
    assert(load == 1);
    wrapper_type test4;
    load = stream_pool::get_current_load<Interface, Pool>();
    assert(load == 2);
    // Check availability method:
    bool avail = stream_pool::interface_available<Interface, Pool>(1);
    assert(avail == false); // NOLINT
    avail = stream_pool::interface_available<Interface, Pool>(2);
    assert(avail == false); // NOLINT
    avail = stream_pool::interface_available<Interface, Pool>(3);
    assert(avail == true); // NOLINT
  }
  auto load0 = stream_pool::get_current_load<Interface, Pool>();
  assert(load0 == 0);
  stream_pool::cleanup<Interface, Pool>();
}

#endif
