// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef STREAM_TEST_HPP // NOLINT
#define STREAM_TEST_HPP // NOLINT
#include <hpx/async_base/apply.hpp>
#include <hpx/async_base/async.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>
#include "cppuddle/memory_recycling/cuda_recycling_allocators.hpp"
#include "cppuddle/memory_recycling/util/cuda_recycling_device_buffer.hpp"
#include "cppuddle/executor_recycling/executor_pools_interface.hpp""

template <typename Interface, typename Pool, typename... Ts>
void test_pool_memcpy(const size_t executor_parameter, Ts &&...ts) {
  std::vector<double,
              cppuddle::memory_recycling::recycle_allocator_cuda_host<double>>
      hostbuffer(512);
  cppuddle::memory_recycling::cuda_device_buffer<double> devicebuffer(512);
  cppuddle::executor_recycling::executor_pool::init<Interface, Pool>(
      executor_parameter, std::forward<Ts>(ts)...);
  // without interface wrapper
  {
    auto test1 =
        cppuddle::executor_recycling::executor_pool::get_interface<Interface,
                                                                   Pool>(0);
    Interface test1_interface = std::get<0>(test1);
    size_t interface_id = std::get<1>(test1);
    hpx::apply(test1_interface, cudaMemcpyAsync,
               devicebuffer.device_side_buffer, hostbuffer.data(),
               512 * sizeof(double), cudaMemcpyHostToDevice);
    auto fut1 = hpx::async(test1_interface, cudaMemcpyAsync, hostbuffer.data(),
                           devicebuffer.device_side_buffer,
                           512 * sizeof(double), cudaMemcpyDeviceToHost);
    fut1.get();
    cppuddle::executor_recycling::executor_pool::release_interface<Interface,
                                                                   Pool>(
        interface_id, 0);
  }

  // with interface wrapper
  {
    cppuddle::executor_recycling::executor_interface<Interface, Pool>
        test1_interface{0};
    // hpx::cuda::cuda_executor test1_interface(0, false);
    hpx::apply(test1_interface.interface, cudaMemcpyAsync,
               devicebuffer.device_side_buffer, hostbuffer.data(),
               512 * sizeof(double), cudaMemcpyHostToDevice);
    auto fut1 = hpx::async(test1_interface.interface, cudaMemcpyAsync,
                           hostbuffer.data(), devicebuffer.device_side_buffer,
                           512 * sizeof(double), cudaMemcpyDeviceToHost);
    fut1.get();
  }
  cppuddle::executor_recycling::executor_pool::cleanup<Interface, Pool>();
}

template <typename Interface, typename Pool, typename... Ts>
void test_pool_ref_counting(const size_t executor_parameter, Ts &&...ts) {

  // init ppol
  cppuddle::executor_recycling::executor_pool::init<Interface, Pool>(
      executor_parameter, std::forward<Ts>(ts)...);
  {
    // Allocating
    auto test1 =
        cppuddle::executor_recycling::executor_pool::get_interface<Interface,
                                                                   Pool>(0);
    auto load1 =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load1 == 0);
    Interface test1_interface = std::get<0>(test1);
    size_t test1_index = std::get<1>(test1);
    auto test2 =
        cppuddle::executor_recycling::executor_pool::get_interface<Interface,
                                                                   Pool>(0);
    auto load2 =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load2 == 1);
    Interface test2_interface = std::get<0>(test2);
    // auto fut = test2_interface.get_future();
    size_t test2_index = std::get<1>(test2);
    auto test3 =
        cppuddle::executor_recycling::executor_pool::get_interface<Interface,
                                                                   Pool>(0);
    auto load3 =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load3 == 1);
    Interface test3_interface = std::get<0>(test3);
    size_t test3_index = std::get<1>(test3);
    auto test4 =
        cppuddle::executor_recycling::executor_pool::get_interface<Interface,
                                                                   Pool>(0);
    auto load4 =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    Interface test4_interface = std::get<0>(test4);
    size_t test4_index = std::get<1>(test4);
    assert(load4 == 2);
    // Releasing
    cppuddle::executor_recycling::executor_pool::release_interface<Interface,
                                                                   Pool>(
        test4_index, 0);
    load4 =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load4 == 1);
    cppuddle::executor_recycling::executor_pool::release_interface<Interface,
                                                                   Pool>(
        test3_index, 0);
    load3 =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load3 == 1);
    cppuddle::executor_recycling::executor_pool::release_interface<Interface,
                                                                   Pool>(
        test2_index, 0);
    load2 =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load2 == 0);
    cppuddle::executor_recycling::executor_pool::release_interface<Interface,
                                                                   Pool>(
        test1_index, 0);
    load1 =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load1 == 0);
  }
  // Clear
  auto load0 =
      cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                    Pool>(0);
  assert(load0 == 0);
  cppuddle::executor_recycling::executor_pool::cleanup<Interface, Pool>();
}

template <typename Interface, typename Pool, typename... Ts>
void test_pool_wrappers(const size_t executor_parameter, Ts &&...ts) {
  using wrapper_type =
      cppuddle::executor_recycling::executor_interface<Interface, Pool>;
  // init ppol
  cppuddle::executor_recycling::executor_pool::init<Interface, Pool>(
      executor_parameter, std::forward<Ts>(ts)...);
  {
    wrapper_type test1{0};
    auto load =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load == 0);
    wrapper_type test2{0};
    load =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    // auto fut = test2.get_future();
    assert(load == 1);
    wrapper_type test3{0};
    load =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load == 1);
    wrapper_type test4{0};
    load =
        cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                      Pool>(0);
    assert(load == 2);
    // Check availability method:
    bool avail =
        cppuddle::executor_recycling::executor_pool::interface_available<
            Interface, Pool>(1, 0);
    assert(avail == false); // NOLINT
    avail = cppuddle::executor_recycling::executor_pool::interface_available<
        Interface, Pool>(2, 0);
    assert(avail == false); // NOLINT
    avail = cppuddle::executor_recycling::executor_pool::interface_available<
        Interface, Pool>(3, 0);
    assert(avail == true); // NOLINT
  }
  auto load0 =
      cppuddle::executor_recycling::executor_pool::get_current_load<Interface,
                                                                    Pool>(0);
  assert(load0 == 0);
  cppuddle::executor_recycling::executor_pool::cleanup<Interface, Pool>();
}

#endif
