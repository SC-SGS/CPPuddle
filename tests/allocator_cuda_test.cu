// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define USE_HPX_MAIN
#include <chrono>
#include <cstdio>
#include <random>
#include <typeinfo>

#include <hpx/async_cuda/cuda_executor.hpp>
#ifdef USE_HPX_MAIN
#include <hpx/hpx_init.hpp>
#else
#include <hpx/hpx_main.hpp>
#endif
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <cuda_runtime.h>

#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"

using executor = hpx::cuda::experimental::cuda_executor;

constexpr size_t N = 200000;
// constexpr size_t chunksize = 20000 ;
constexpr size_t chunksize = 20000;
constexpr size_t passes = 10;

__global__ void hello(void) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%i", i);
}

__global__ void mv(const size_t startindex, const size_t chunksize,
                   const size_t N, double *y, double *erg) {
  // Matrix is row major
  int item_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (item_index < chunksize) {
    double tmp = 0.0;
    int start_row = (startindex + item_index) * N;
    for (size_t i = 0; i < N; i++) {
      double a = (i + start_row) % 2;
      tmp += a * y[i];
    }
    erg[item_index] = tmp;
  }
}

// #pragma nv_exec_check_disable
#ifdef USE_HPX_MAIN
int hpx_main(int argc, char *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
  executor cuda_interface(0, false); // one stream per HPX thread
  // dim3 const grid_spec(1, 1, 1);
  // dim3 const threads_per_block(1, 1, 1);
  // void *args[] = {};
  // std::cout << " starting kernel..." << std::endl;
  // auto fut =
  //     cuda_interface.async_execute(cudaLaunchKernel<decltype(hello)>, hello,
  //                                  grid_spec, threads_per_block, args, 0);
  // std::cout << " kernel started..." << std::endl;
  // fut.get();
  // std::cout << " kernel finished..." << std::endl;
  // std::cin.get();

  // Generate Problem: Repeated (unpotizmized) Matrix-Vec
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  std::vector<double> mult_vector(N + chunksize);
  std::vector<double> erg_vector(N + chunksize);
  for (auto &entry : mult_vector) {
    entry = dis(gen);
  }

  // Nonensical function, used to simulate stuff happening on the cpu
  auto cpu_side_function = [](auto &partial_result, size_t chunksize) {
    for (size_t i = 0; i < chunksize; i++)
      partial_result[i] *= 0.5;
  };

  // Begin iterations
  constexpr size_t number_packages =
      N / chunksize + static_cast<int>(N % chunksize);
  size_t problemsize =
      N; // interface.execute does not like constexpr so we got these
  size_t chunk = chunksize;
  std::array<hpx::shared_future<void>, number_packages> futs;
  for (size_t i = 0; i < number_packages; i++) {
    futs[i] = hpx::make_ready_future<void>();
  }

  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t pass = 0; pass < passes; pass++) {
    // Divide into work packages - Create tasks
    size_t fut_index = 0;
    for (size_t row_index = 0; row_index < N;
         row_index += chunksize, fut_index++) {
      futs[fut_index] = futs[fut_index].then(
          [row_index, &erg_vector, mult_vector, &problemsize, &chunk,
           &cpu_side_function](hpx::shared_future<void> &&f) {
            // Recycle communication channels
            executor cuda_interface(0, false);
            std::vector<double, recycler::recycle_allocator_cuda_host<double>>
                input_host(N + 128);
            recycler::cuda_device_buffer<double> input_device(N + 128);
            std::vector<double, recycler::recycle_allocator_cuda_host<double>>
                erg_host(N + 128);
            recycler::cuda_device_buffer<double> erg_device(N + 128);
            // Copy into input array
            for (size_t i = 0; i < chunksize; i++) {
              input_host[i] = mult_vector[row_index + i];
            }
            cuda_interface.post(
                cudaMemcpyAsync, input_device.device_side_buffer,
                input_host.data(), N * sizeof(double), cudaMemcpyHostToDevice);
            size_t row = row_index;
            dim3 const grid_spec((chunksize + 127) / 128, 1, 1);
            dim3 const threads_per_block(128, 1, 1);
            void *args[] = {&row, &chunk, &problemsize,
                            &(input_device.device_side_buffer),
                            &(erg_device.device_side_buffer)};
            cuda_interface.post(cudaLaunchKernel<decltype(mv)>, mv, grid_spec,
                                threads_per_block, args, 0);
            auto fut = cuda_interface.async_execute(
                cudaMemcpyAsync, erg_host.data(), erg_device.device_side_buffer,
                N * sizeof(double), cudaMemcpyHostToDevice);
            fut.get();
            // To CPU side function
            cpu_side_function(erg_host, chunk);
            for (size_t i = 0; i < chunksize; i++) {
              erg_vector[row_index + i] = input_host[i];
            }
          });
    }
  }
  auto when = hpx::when_all(futs);
  when.wait();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "\n==>Mults took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "ms" << std::endl;

  recycler::force_cleanup(); // depending on the driver we cannot rely on the cleanup during the static exit time
                             // as the cuda runtime may already be unloaded
  return hpx::finalize();
}

#ifdef USE_HPX_MAIN
int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
#endif
