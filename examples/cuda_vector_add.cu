#include <algorithm>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>


#include <cppuddle/memory_recycling/std_recycling_allocators.hpp>
#include <cppuddle/memory_recycling/cuda_recycling_allocators.hpp>
#include <cppuddle/memory_recycling/util/cuda_recycling_device_buffer.hpp>
#include <cppuddle/executor_recycling/executor_pools_interface.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

using float_t = float;
using device_executor_t = hpx::cuda::experimental::cuda_executor;

constexpr size_t vector_size = 102400;
constexpr size_t entries_per_task = 1024;
constexpr size_t number_tasks = vector_size / entries_per_task;
constexpr size_t number_repetitions = 20;
constexpr size_t max_queue_length = 5;
constexpr size_t number_executors = 1;
constexpr size_t gpu_id = 0;

static_assert(vector_size % entries_per_task == 0);


__global__ void kernel_add(const float_t *input_a, const float_t *input_b, float_t *output_c) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  output_c[index] = input_a[index] + input_b[index];
}

int hpx_main(int argc, char *argv[]) {
  // HPX and CPPuddle Setup for executor (polling + pool init)
  // =========================================== 0.a Init HPX CUDA polling
  hpx::cout << "Start initializing CUDA polling and executor pool..." << std::endl;
  hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
  // 0.b Init CPPuddle executor pool
  cppuddle::executor_recycling::executor_pool::init_executor_pool<
      device_executor_t,
      cppuddle::executor_recycling::round_robin_pool_impl<device_executor_t>>(
      gpu_id, number_executors, gpu_id, true);
  hpx::cout << "Init done!" << std::endl << std::endl;

  std::atomic<size_t> number_cpu_kernel_launches = 0;
  std::atomic<size_t> number_gpu_kernel_launches = 0;

  // Launch tasks 
  // Note: Repetitions may be out of order since they do not depend on each other in this toy sample
  hpx::cout << "Start launching tasks..." << std::endl;
  std::vector<hpx::future<void>> repetition_futs(number_repetitions);
  for (size_t repetition = 0; repetition < number_repetitions; repetition++) {
    std::vector<hpx::future<void>> futs(number_tasks);
    for (size_t task_id = 0; task_id < number_tasks; task_id++) {
      futs[task_id] = hpx::async([task_id, &number_cpu_kernel_launches,
                                  &number_gpu_kernel_launches]() {
        // Inner Task Setup to launch the CUDA kernels:
        // ===========================================

        // 1. Create required per task host-side buffers
        std::vector<
            float_t,
            cppuddle::memory_recycling::recycle_allocator_cuda_host<float_t>>
            host_a(entries_per_task);
        std::vector<
            float_t,
            cppuddle::memory_recycling::recycle_allocator_cuda_host<float_t>>
            host_b(entries_per_task);
        std::vector<
            float_t,
            cppuddle::memory_recycling::recycle_allocator_cuda_host<float_t>>
            host_c(entries_per_task);

        // 2. Host-side preprocessing (usually: communication, here fill dummy
        // input)
        std::fill(host_a.begin(), host_a.end(), 1.0);
        std::fill(host_b.begin(), host_b.end(), 2.0);

        // 3. Check GPU utiliation
        bool device_executor_available =
            cppuddle::executor_recycling::executor_pool::interface_available<
                device_executor_t,
                cppuddle::executor_recycling::round_robin_pool_impl<
                    device_executor_t>>(max_queue_length, gpu_id);

        //4. Run Kernel on either CPU or GPU
        if (!device_executor_available) {
          // 4a. Launch CPU Fallback  Version
          number_cpu_kernel_launches++;
          for (size_t entry_id = 0; entry_id < entries_per_task; entry_id++) {
            host_c[entry_id] = host_a[entry_id] + host_b[entry_id];
          }
        } else {
          // 4b. Create per_task device-side buffers and draw executor
          number_gpu_kernel_launches++;
          cppuddle::executor_recycling::executor_interface<
              device_executor_t, cppuddle::executor_recycling::
                                     round_robin_pool_impl<device_executor_t>>
              executor(gpu_id);
          cppuddle::memory_recycling::cuda_device_buffer<float_t>
              device_a(entries_per_task);
          cppuddle::memory_recycling::cuda_device_buffer<float_t>
              device_b(entries_per_task);
          cppuddle::memory_recycling::cuda_device_buffer<float_t>
              device_c(entries_per_task);

          // 4c. Launch data transfers and kernel
          hpx::apply(
              static_cast<device_executor_t>(executor),
              cudaMemcpyAsync, device_a.device_side_buffer, host_a.data(),
              entries_per_task * sizeof(float_t), cudaMemcpyHostToDevice);
          hpx::apply(
              static_cast<device_executor_t>(executor),
              cudaMemcpyAsync, device_b.device_side_buffer, host_b.data(),
              entries_per_task * sizeof(float_t), cudaMemcpyHostToDevice);
          void *args[] = {&device_a.device_side_buffer,
                          &device_b.device_side_buffer,
                          &device_c.device_side_buffer};
          hpx::apply(
              static_cast<device_executor_t>(executor),
              cudaLaunchKernel<decltype(kernel_add)>, kernel_add,
              entries_per_task / 128, 128, args, 0);
          auto fut = hpx::async(
              static_cast<device_executor_t>(executor),
              cudaMemcpyAsync, host_c.data(), device_c.device_side_buffer,
              entries_per_task * sizeof(float_t), cudaMemcpyDeviceToHost);
          fut.get(); // Allow worker thread to jump away until the kernel and
                     // data-transfers are done
        }

        // 5. Host-side postprocessing (usually: communication, here: check
        // correctness)
        if (!std::all_of(host_c.begin(), host_c.end(),
                         [](float_t i) { return i == 1.0 + 2.0; })) {
          std::cerr << "Task " << task_id << " contained wrong results!!"
                    << std::endl;
        }

        // Inner Task Done!
        // ===========================================
      });
    }
    // Schedule output task to run once a repetition is done
    auto repetition_finished = hpx::when_all(futs);
    repetition_futs.emplace_back(repetition_finished.then([repetition](auto &&fut) {
            hpx::cout << "Repetition " << repetition << " done!" << std::endl;
          }));
  }
  hpx::cout << "All tasks launched asynchronously!" << std::endl << std::endl;
  // Schedule output task to run once all other tasks are done
  auto all_done_fut =
      hpx::when_all(repetition_futs)
          .then([&number_cpu_kernel_launches,
                 &number_gpu_kernel_launches](auto &&fut) {
            hpx::cout << "All tasks are done!" << std::endl;
            hpx::cout << " => " << number_gpu_kernel_launches
                      << " kernels were run on the GPU" << std::endl;
            hpx::cout << " => " << number_cpu_kernel_launches
                      << " kernels were using the CPU fallback" << std::endl << std::endl;
          });
  all_done_fut.get();

  hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
  hpx::cout << "Finalizing..." << std::endl;
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
