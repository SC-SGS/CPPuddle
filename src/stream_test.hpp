#ifndef STREAM_TEST_HPP // NOLINT
#define STREAM_TEST_HPP // NOLINT
#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"

template <typename Interface, typename Pool>
void test_pool_memcpy(const size_t gpu_parameter,
                      const size_t stream_parameter) {
  std::vector<double, recycler::recycle_allocator_cuda_host<double>> hostbuffer(
      512);
  recycler::cuda_device_buffer<double> devicebuffer(512);
  stream_pool::init<Interface, Pool>(gpu_parameter, stream_parameter);
  {
    auto test1 = stream_pool::get_interface<Interface, Pool>();
    Interface test1_interface = std::get<0>(test1);
    test1_interface.copy_async(devicebuffer.device_side_buffer,
                               hostbuffer.data(), 512 * sizeof(double),
                               cudaMemcpyHostToDevice);
    test1_interface.copy_async(hostbuffer.data(),
                               devicebuffer.device_side_buffer,
                               512 * sizeof(double), cudaMemcpyDeviceToHost);
    auto fut1 = test1_interface.get_future();
    fut1.get();
  }

  {
    stream_interface<Interface, Pool> test1_interface(0);
    test1_interface.copy_async(devicebuffer.device_side_buffer,
                               hostbuffer.data(), 512 * sizeof(double),
                               cudaMemcpyHostToDevice);
    test1_interface.copy_async(hostbuffer.data(),
                               devicebuffer.device_side_buffer,
                               512 * sizeof(double), cudaMemcpyDeviceToHost);
    auto fut1 = test1_interface.get_future();
    fut1.get();
  }
}

template <typename Interface, typename Pool>
void test_pool_ref_counting(const size_t gpu_parameter,
                            size_t stream_parameter) {

  // init ppol
  stream_pool::init<Interface, Pool>(gpu_parameter, stream_parameter);
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
    auto fut = test2_interface.get_future();
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
}

template <typename Interface, typename Pool>
void test_pool_wrappers(const size_t gpu_parameter, size_t stream_parameter) {
  using wrapper_type = stream_interface<Interface, Pool>;
  // init ppol
  stream_pool::init<Interface, Pool>(gpu_parameter, stream_parameter);
  {
    wrapper_type test1(0);
    auto load = stream_pool::get_current_load<Interface, Pool>();
    assert(load == 0);
    wrapper_type test2(0);
    load = stream_pool::get_current_load<Interface, Pool>();
    auto fut = test2.get_future();
    assert(load == 1);
    wrapper_type test3(0);
    load = stream_pool::get_current_load<Interface, Pool>();
    assert(load == 1);
    wrapper_type test4(0);
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
}

#endif