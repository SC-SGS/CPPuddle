#include <algorithm>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>

#include <boost/program_options.hpp>

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
constexpr bool in_order_repetitions = true;

static_assert(vector_size % entries_per_task == 0);


__global__ void kernel_add(const float_t *input_a, const float_t *input_b, float_t *output_c) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  output_c[index] = input_a[index] + input_b[index];
}

int hpx_main(int argc, char *argv[]) {

  /* try { */
  /*   boost::program_options::options_description desc{"Options"}; */
  /*   desc.add_options()("help", "Help screen")( */
  /*       "elements_per_task", */
  /*       boost::program_options::value<size_t>(&array_size) */
  /*           ->default_value(5000000), */
  /*       "Size of the buffers")( */
  /*       "tasks_per_repetition", */
  /*       boost::program_options::value<size_t>(&number_futures) */
  /*           ->default_value(64), */
  /*       "Sets the number of futures to be (potentially) executed in parallel")( */
  /*       "number_repetitions", */
  /*       boost::program_options::value<size_t>(&passes)->default_value(200), */
  /*       "Sets the number of repetitions")( */
  /*       "outputfile", */
  /*       boost::program_options::value<std::string>(&filename)->default_value( */
  /*           ""), */
  /*       "Redirect stdout/stderr to this file"); */

  /*   boost::program_options::variables_map vm; */
  /*   boost::program_options::parsed_options options = */
  /*       parse_command_line(argc, argv, desc); */
  /*   boost::program_options::store(options, vm); */
  /*   boost::program_options::notify(vm); */

  /*   if (vm.count("help") == 0u) { */
  /*     std::cout << "Running with parameters:" << std::endl */
  /*               << " --arraysize = " << array_size << std::endl */
  /*               << " --futures =  " << number_futures << std::endl */
  /*               << " --passes = " << passes << std::endl */
  /*               << " --hpx:threads = " << hpx::get_os_thread_count() */
  /*               << std::endl; */
  /*   } else { */
  /*     std::cout << desc << std::endl; */
  /*     return hpx::finalize(); */
  /*   } */
  /* } catch (const boost::program_options::error &ex) { */
  /*   std::cerr << "CLI argument problem found: " << ex.what() << '\n'; */
  /* } */

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

  hpx::shared_future<void> previous_iteration_fut = hpx::make_ready_future<void>();
  std::vector<hpx::future<void>> repetition_futs(number_repetitions);
  for (size_t repetition = 0; repetition < number_repetitions; repetition++) {
    std::vector<hpx::future<void>> futs(number_tasks);
    for (size_t task_id = 0; task_id < number_tasks; task_id++) {
      auto gpu_task_lambda = [](const auto task_id,
                                auto &number_cpu_kernel_launches,
                                auto &number_gpu_kernel_launches) {
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
      };
      if (in_order_repetitions) {
        futs[task_id] =  previous_iteration_fut.then([task_id, &number_cpu_kernel_launches,
                        &number_gpu_kernel_launches, gpu_task_lambda](auto && fut) {
              gpu_task_lambda(task_id, number_cpu_kernel_launches,
                              number_gpu_kernel_launches);
            });
      } else {
        futs[task_id] =
            hpx::async([task_id, &number_cpu_kernel_launches,
                        &number_gpu_kernel_launches, gpu_task_lambda]() {
              gpu_task_lambda(task_id, number_cpu_kernel_launches,
                              number_gpu_kernel_launches);
            });
      }
    }
    // Schedule output task to run once a repetition is done
    auto repetition_finished = hpx::when_all(futs);
    if (in_order_repetitions) {
      previous_iteration_fut =
          repetition_finished.then([repetition](auto &&fut) {
            hpx::cout << "Repetition " << repetition << " done!" << std::endl;
          });
    } else {
      repetition_futs.emplace_back(
          repetition_finished.then([repetition](auto &&fut) {
            hpx::cout << "Repetition " << repetition << " done!" << std::endl;
          }));
    }
  }
  hpx::cout << "All tasks launched asynchronously!" << std::endl << std::endl;
  // Schedule output task to run once all other tasks are done
  if (in_order_repetitions) {
    previous_iteration_fut
        .then([&number_cpu_kernel_launches,
               &number_gpu_kernel_launches](auto &&fut) {
          hpx::cout << "All tasks are done! [in-order repetitions version]" << std::endl;
          hpx::cout << " => " << number_gpu_kernel_launches
                    << " kernels were run on the GPU" << std::endl;
          hpx::cout << " => " << number_cpu_kernel_launches
                    << " kernels were using the CPU fallback" << std::endl
                    << std::endl;
        })
        .get();
  } else {
    hpx::when_all(repetition_futs)
        .then([&number_cpu_kernel_launches,
               &number_gpu_kernel_launches](auto &&fut) {
          hpx::cout << "All tasks are done! [out-of-order repetitions version]" << std::endl;
          hpx::cout << " => " << number_gpu_kernel_launches
                    << " kernels were run on the GPU" << std::endl;
          hpx::cout << " => " << number_cpu_kernel_launches
                    << " kernels were using the CPU fallback" << std::endl
                    << std::endl;
        })
        .get();
  }

  hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
  hpx::cout << "Finalizing..." << std::endl;
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
