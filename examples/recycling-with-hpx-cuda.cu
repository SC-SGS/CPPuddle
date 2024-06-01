// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Developer  TODOs regarding CPPuddle usability:
// TODO(daissgr) Simplify specifying an executor pool (at least when using the
// default round_robin_pool_impl). The current way seems awfully verbose

#include <algorithm>
#include <cstdlib>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>

#include <boost/program_options.hpp>

#include <cppuddle/memory_recycling/buffer_management_interface.hpp>
#include <cppuddle/memory_recycling/std_recycling_allocators.hpp>
#include <cppuddle/memory_recycling/cuda_recycling_allocators.hpp>
#include <cppuddle/memory_recycling/util/cuda_recycling_device_buffer.hpp>
#include <cppuddle/executor_recycling/executor_pools_interface.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>


/** \file This example shows how to use HPX + CPPuddle with GPU-accelerated
 * applications. Particulary we focus on how to use a) recycled pinned host
 * memory, b) recycled device memory, c) the executor pool, d) the HPX-CUDA
 * futures and the basic CPU/GPU load balancing based on executor usage in an
 * HPX application. To demonstrate these features we just use the simplest of
 * kernels: a vector addition that is repeated over a multitude of tasks (with
 * varying, artifical dependencies inbetween). So while the compute kernel is
 * basic, we still get to see how the CPPuddle/HPX features may be used with 
 * it.
 *
 * The example has three parts: First the GPU part, then the HPX task graph
 * management and lastly the remaining initialization/boilerplate code
 */

//=================================================================================================
// PART I: The (CUDA) GPU kernel and how to launch it with CPPuddle + HPX whilst avoid
// any CPU/GPU barriers
//=================================================================================================

// Compile-time options: float type...
using float_t = float;
// ... and we will use the HPX CUDA executor inside the executor pool later on
using device_executor_t = hpx::cuda::experimental::cuda_executor;

/** Just some example CUDA kernel. For simplicity it just adds two vectors. */
__global__ void kernel_add(const float_t *input_a, const float_t *input_b, float_t *output_c) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  output_c[index] = input_a[index] + input_b[index];
}

/** Method that demonstrates how one might launch a CUDA kernel with HPX and
 * CPPuddle recycled memory/executor! By using CPPuddle allocators to avoid
 * allocating GPU memory and HPX futures to track the status of the
 * kernel/memory transfers, this method is expected to be non-blocking both on
 * the launching CPU thread and on the GPU (non malloc barriers). Hence, this
 * launch method is suitable to quickly launch a multitude of GPU kernels if
 * required.
 * 
 * This method uses the following features:
 * - Recycled pinned host memory.
 * - Recycled device memory.
 * - Draws GPU executor from the CPPuddle executor pool.
 * - CPU-GPU load balancing based on the number of GPU executors and their queue  length.
 * - Asynchronous data-transfers and lauching of the kernel.
 * - HPX futures to suspend the HPX task until kernel and data-transfers are  done.
 * - Includes (sample) pre- and post-processing. */
void launch_gpu_kernel_task(const size_t task_id, const size_t entries_per_task,
                            const size_t max_queue_length, const size_t gpu_id,
                            std::atomic<size_t> &number_cpu_kernel_launches,
                            std::atomic<size_t> &number_gpu_kernel_launches) {
  // 1. Create required per task host-side buffers using CPPuddle recycled
  // pinned memory
  std::vector<float_t,
              cppuddle::memory_recycling::recycle_allocator_cuda_host<float_t>>
      host_a(entries_per_task);
  std::vector<float_t,
              cppuddle::memory_recycling::recycle_allocator_cuda_host<float_t>>
      host_b(entries_per_task);
  std::vector<float_t,
              cppuddle::memory_recycling::recycle_allocator_cuda_host<float_t>>
      host_c(entries_per_task);

  // 2. Host-side preprocessing (usually: communication, here fill dummy input)
  std::fill(host_a.begin(), host_a.end(), 1.0);
  std::fill(host_b.begin(), host_b.end(), 2.0);

  // 3. Check GPU utilization - Method will return true if there is an executor
  // in the pool that does currently not exceed its queue limit (tracked by
  // RAII, no CUDA API calls involved)
  bool device_executor_available =
      cppuddle::executor_recycling::executor_pool::interface_available<
          device_executor_t, cppuddle::executor_recycling::
                                 round_robin_pool_impl<device_executor_t>>(
          max_queue_length, gpu_id);

  // 4. Run Kernel on either CPU or GPU
  if (!device_executor_available) {
    number_cpu_kernel_launches++;
    // 4a. Launch CPU Fallback  Version
    for (size_t entry_id = 0; entry_id < entries_per_task; entry_id++) {
      host_c[entry_id] = host_a[entry_id] + host_b[entry_id];
    }
  } else {
    number_gpu_kernel_launches++;
    // 4b. Create per_task device-side buffers (using recylced device memory)
    // and draw GPU executor from CPPuddle executor pool
    cppuddle::executor_recycling::executor_interface<
        device_executor_t,
        cppuddle::executor_recycling::round_robin_pool_impl<device_executor_t>>
        executor(gpu_id); // Wrapper that draws executor from the pool
    cppuddle::memory_recycling::cuda_device_buffer<float_t> device_a(
        entries_per_task);
    cppuddle::memory_recycling::cuda_device_buffer<float_t> device_b(
        entries_per_task);
    cppuddle::memory_recycling::cuda_device_buffer<float_t> device_c(
        entries_per_task);

    // 4c. Launch data transfers and kernel
    hpx::apply(static_cast<device_executor_t>(executor), cudaMemcpyAsync,
               device_a.device_side_buffer, host_a.data(),
               entries_per_task * sizeof(float_t), cudaMemcpyHostToDevice);
    hpx::apply(static_cast<device_executor_t>(executor), cudaMemcpyAsync,
               device_b.device_side_buffer, host_b.data(),
               entries_per_task * sizeof(float_t), cudaMemcpyHostToDevice);
    void *args[] = {&device_a.device_side_buffer, &device_b.device_side_buffer,
                    &device_c.device_side_buffer};
    hpx::apply(static_cast<device_executor_t>(executor),
               cudaLaunchKernel<decltype(kernel_add)>, kernel_add,
               entries_per_task / 128, 128, args, 0);
    auto fut =
        hpx::async(static_cast<device_executor_t>(executor), cudaMemcpyAsync,
                   host_c.data(), device_c.device_side_buffer,
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
}

//=================================================================================================
// PART II: How to build the dependency graph with HPX and the GPU launches
//=================================================================================================

/** This methods demonstrates how one might build the HPX task graph
 * asynchronously, using the launch_gpu_kernel_task method to launch the GPU
 * kernels inside the tasks. To illustrate how one can chain together tasks, we
 * support two modes for building the task tree: One keeps the dependencies
 * between the repetitions (keeping them in order) and one does not and allows
 * to interleave repetitions. */
hpx::future<void>
build_task_graph(const size_t number_repetitions, const size_t number_tasks,
                 const size_t entries_per_task, const bool in_order_repetitions,
                 const size_t max_queue_length, const size_t gpu_id,
                 std::atomic<size_t> &number_cpu_kernel_launches,
                 std::atomic<size_t> &number_gpu_kernel_launches) {
  // Launch tasks
  hpx::shared_future<void> previous_iteration_fut = hpx::make_ready_future<void>();
  std::vector<hpx::future<void>> repetition_futs(number_repetitions);
  for (size_t repetition = 0; repetition < number_repetitions; repetition++) {
    std::vector<hpx::future<void>> futs(number_tasks);
    for (size_t task_id = 0; task_id < number_tasks; task_id++) {
      // Schedule task either in order (one repetition after another) or out of order
      if (in_order_repetitions) {
        futs[task_id] = previous_iteration_fut.then(
            [task_id, entries_per_task, max_queue_length, gpu_id,
             &number_cpu_kernel_launches,
             &number_gpu_kernel_launches](auto &&fut) {
              launch_gpu_kernel_task(
                  task_id, entries_per_task, max_queue_length, gpu_id,
                  number_cpu_kernel_launches, number_gpu_kernel_launches);
            });
      } else {
        futs[task_id] = hpx::async([task_id, entries_per_task, max_queue_length,
                                    gpu_id, &number_cpu_kernel_launches,
                                    &number_gpu_kernel_launches]() {
          launch_gpu_kernel_task(task_id, entries_per_task, max_queue_length,
                                 gpu_id, number_cpu_kernel_launches,
                                 number_gpu_kernel_launches);
        });
      }
    }
    // Schedule output task to run once each repetition is done
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
  // Schedule final output task to run once all other tasks are done and return future
  if (in_order_repetitions) {
    return previous_iteration_fut
        .then([&number_cpu_kernel_launches,
               &number_gpu_kernel_launches](auto &&fut) {
          hpx::cout << "All tasks are done! [in-order repetitions version]" << std::endl;
          hpx::cout << " => " << number_gpu_kernel_launches
                    << " kernels were run on the GPU" << std::endl;
          hpx::cout << " => " << number_cpu_kernel_launches
                    << " kernels were using the CPU fallback" << std::endl
                    << std::endl;
        });
  } else {
    return hpx::when_all(repetition_futs)
        .then([&number_cpu_kernel_launches,
               &number_gpu_kernel_launches](auto &&fut) {
          hpx::cout << "All tasks are done! [out-of-order repetitions version]" << std::endl;
          hpx::cout << " => " << number_gpu_kernel_launches
                    << " kernels were run on the GPU" << std::endl;
          hpx::cout << " => " << number_cpu_kernel_launches
                    << " kernels were using the CPU fallback" << std::endl
                    << std::endl;
        });
  }
}

//=================================================================================================
// PART III: Initialization / Boilerplate and Main
//=================================================================================================

/** HPX uses either callbacks or event polling to implement its CUDA futures.
 * Polling usually has the superior performance, however, it requires that the
 * polling is initialized at startup (or at least before the CUDA futures are
 * used). The CPPuddle executor pool also needs initialzing as we need to set it
 * to a specified number of executors (which CPPuddle cannot know without the
 * number_gpu_executors parameter). We will use the round_robin_pool_impl for
 * simplicity. A priority_pool_impl is also available.
 */
void init_executor_pool_and_polling(const size_t number_gpu_executors, const size_t gpu_id) {
  assert(gpu_id == 0); // MultiGPU not used in this example
  hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
  cppuddle::executor_recycling::executor_pool::init_executor_pool<
      device_executor_t,
      cppuddle::executor_recycling::round_robin_pool_impl<device_executor_t>>(
      gpu_id, number_gpu_executors, gpu_id, true);
}

/// Processes the CLI options via boost program_options to configure the example
bool process_cli_options(int argc, char *argv[], size_t &entries_per_task,
                         size_t &number_tasks, bool &in_order_repetitions,
                         size_t &number_repetitions, size_t &number_gpu_executors,
                         size_t &max_queue_length) {
  try {
    boost::program_options::options_description desc{"Options"};
    desc.add_options()("help", "Help screen")(
        "elements_per_task",
        boost::program_options::value<size_t>(&entries_per_task)
            ->default_value(1024),
        "Number of elements added per task (corresponds to the number of CUDA "
        "workitems used per kernel)")(
        "tasks_per_repetition",
        boost::program_options::value<size_t>(&number_tasks)
            ->default_value(100),
        "Number of tasks per repetition")(
        "in_order_repetitions",
        boost::program_options::value<bool>(&in_order_repetitions)
            ->default_value(false),
        "Execute repetitions in-order")(
        "number_repetitions",
        boost::program_options::value<size_t>(&number_repetitions)
            ->default_value(20),
        "Sets the number of repetitions")(
        "number_gpu_executors",
        boost::program_options::value<size_t>(&number_gpu_executors)
            ->default_value(32),
        "Number of GPU executors in the pool")(
        "max_queue_length_per_executor",
        boost::program_options::value<size_t>(&max_queue_length)
            ->default_value(5),
        "Maximum numbers of kernels queued per GPU executor");

    boost::program_options::variables_map vm;
    boost::program_options::parsed_options options =
        parse_command_line(argc, argv, desc);
    boost::program_options::store(options, vm);
    boost::program_options::notify(vm);

    if (entries_per_task % 128 != 0) {
      std::cerr << "ERROR: --entries_per_task needs to be divisble by 128." << std::endl;
      return false;
    }

    std::cout << "CPPuddle Recycling Sample (Vector-Add / CUDA edition)" << std::endl;
    std::cout << "=====================================================" << std::endl;
    if (vm.count("help") == 0u) {
      hpx::cout << "Running with parameters:" << std::endl
                << " --elements_per_task = " << entries_per_task << std::endl
                << " --tasks_per_repetition =  " << number_tasks << std::endl
                << " --number_repetitions = " << number_repetitions << std::endl
                << " --in_order_repetitions = " << in_order_repetitions << std::endl
                << " --number_gpu_executors = " << number_gpu_executors << std::endl
                << " --max_queue_length_per_executor = " << max_queue_length << std::endl
                << " --hpx:threads = " << hpx::get_os_thread_count()
                << std::endl << std::endl;
    } else {
      std::cout << desc << std::endl;
      return false;
    }
  } catch (const boost::program_options::error &ex) {
    std::cerr << "CLI argument problem found: " << ex.what() << '\n';
    return false;
  }
  return true;
}

int hpx_main(int argc, char *argv[]) {
  // Launch counters
  std::atomic<size_t> number_cpu_kernel_launches = 0;
  std::atomic<size_t> number_gpu_kernel_launches = 0;

  // Runtime options
  size_t entries_per_task = 1024;
  size_t number_tasks = 100;
  size_t number_repetitions = 20;
  bool in_order_repetitions = false;
  size_t max_queue_length = 5;
  size_t number_gpu_executors = 1;
  size_t gpu_id = 0;
  if(!process_cli_options(argc, argv, entries_per_task, number_tasks,
                      in_order_repetitions, number_repetitions,
                      number_gpu_executors, max_queue_length)) {
    return hpx::finalize(); // problem with CLI parameters detected -> exiting..
  }

  // Init HPX CUDA polling + executor pool
  hpx::cout << "Start initializing CUDA polling and executor pool..." << std::endl;
  init_executor_pool_and_polling(number_gpu_executors, gpu_id);
  hpx::cout << "Init done!" << std::endl << std::endl;


  // Build task graph / Launch all tasks
  auto start = std::chrono::high_resolution_clock::now();
  hpx::cout << "Start launching tasks..." << std::endl;
  auto all_tasks_done_fut =
      build_task_graph(number_repetitions, number_tasks, entries_per_task,
                       in_order_repetitions, max_queue_length, gpu_id,
                       number_cpu_kernel_launches, number_gpu_kernel_launches);
  hpx::cout << "All tasks launched asynchronously!" << std::endl; 
  // Only continue once all tasks are done!
  all_tasks_done_fut.get();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  hpx::cout << "Launching and running all tasks took " << microseconds
            << " microseconds!" << std::endl
            << std::endl;

  // Finalize HPX (CPPuddle finalizes automatically)
  hpx::cout << "Finalizing..." << std::endl;
  // Deallocates all CPPuddle everything and prevent further usage. Technically
  // not required as long as static variables with CPPuddle-managed memory are
  // not used, however, it does not hurt either.
  cppuddle::memory_recycling::finalize();
  hpx::cuda::experimental::detail::unregister_polling(
      hpx::resource::get_thread_pool(0));
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
