#ifndef STREAM_TEST_HPP
#define STREAM_TEST_HPP
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

#endif