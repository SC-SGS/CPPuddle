#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include "../include/buffer_manager.hpp"
#include <cstdio>
#include <typeinfo>
#include <chrono>


constexpr size_t number_futures = 64;
constexpr size_t array_size = 1000000;
constexpr size_t passes = 10;

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{
  std::vector<float, recycle_allocator_cuda_host<float>> vector1_h(array_size);
  auto init_fut1 = hpx::async([&]() {
    for (size_t i = 0; i < array_size; i++) {
      vector1_h[i] = 1.0 + i%8;
    }
  });
  cuda_device_buffer<float> vector1_d(array_size);
  auto launch_move_fut1 = init_fut1.then([&](hpx::future<void> &&f) {
    cudaMemcpy(vector1_d.device_side_buffer, vector1_h.data(), array_size * sizeof(float), cudaMemcpyHostToDevice);
  });

  std::vector<float, recycle_allocator_cuda_host<float>> vector2_h(array_size);
  auto init_fut2 = hpx::async([&]() {
    for (size_t i = 0; i < array_size; i++) {
      vector2_h[i] = 1.0 + i%8;
    }
  });
  cuda_device_buffer<float> vector2_d(array_size);
  auto launch_move_fut2 = init_fut2.then([&](hpx::future<void> &&f) {
    cudaMemcpy(vector2_d.device_side_buffer, vector2_h.data(), array_size * sizeof(float), cudaMemcpyHostToDevice);
  });

  std::vector<float, recycle_allocator_cuda_host<float>> vector3_h(array_size);
  auto init_fut3 = hpx::async([&]() {
    for (size_t i = 0; i < array_size; i++) {
      vector3_h[i] = 1.0 + i%8;
    }
  });
  cuda_device_buffer<float> vector3_d(array_size);
  auto launch_move_fut3 = init_fut3.then([&](hpx::future<void> &&f) {
    cudaMemcpy(vector3_d.device_side_buffer, vector3_h.data(), array_size * sizeof(float), cudaMemcpyHostToDevice);
  });

  std::vector<float, recycle_allocator_cuda_host<float>> result_h(array_size);
  cuda_device_buffer<float> result_d(array_size);

  auto when = hpx::when_all(launch_move_fut1, launch_move_fut2, launch_move_fut3);
  when.wait();

  saxpy<<<(array_size+255)/256, 256>>>(array_size, 2.0f, vector1_d.device_side_buffer, result_d.device_side_buffer);

  cudaMemcpy(result_h.data(), result_d.device_side_buffer, array_size*sizeof(float), cudaMemcpyDeviceToHost);
  for (auto elem : result_h) {
    std::cout << elem << " ";
  }

}
