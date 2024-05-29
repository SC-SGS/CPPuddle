// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Developer  TODOs regarding CPPuddle usability:
// TODO(daissgr) Improve type accessiblity (user should not worry about the
// activated Kokkos backend like belew to pick the correct view types
// TODO(daissgr) Add unified CPPuddle finalize that also cleans up all executor
// pool (and avoids having to use the cleanup methds of the individual pools

#include <algorithm>
#include <cstdlib>

#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>

#include <hpx/kokkos.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <cppuddle/memory_recycling/buffer_management_interface.hpp>
#include <cppuddle/memory_recycling/std_recycling_allocators.hpp>
#include <cppuddle/memory_recycling/cuda_recycling_allocators.hpp>
#include <cppuddle/memory_recycling/util/recycling_kokkos_view.hpp>
#include <cppuddle/executor_recycling/executor_pools_interface.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>


/** \file This example shows how to use HPX + Kokkos + CPPuddle with GPU-accelerated
 * applications. The example is extremly similary to its CUDA counterpart, however, uses
 * Kokkos for implementation to showcase the required boilerplate and offered features.
 * Particulary we focus on how to use a) recycled pinned host
 * memory, b) recycled device memory, c) the executor pool, d) the HPX-Kokkos
 * futures and the basic CPU/GPU load balancing based on executor usage in an
 * HPX application. To demonstrate these features we just use the simplest of
 * kernels: a vector add, that is repeated over a multitude of tasks (with
 * varying, artifical dependencies inbetween). So while the compute kernel is
 * basic, we still get to see how the CPPuddle/HPX features may be used.
 *
 * The example has three parts: First the GPU part, then the HPX task graph
 * management and lastly the remaining initialization/boilerplate code
 */

//=================================================================================================
// PART I: The Kokkos kernel and how to launch it with CPPuddle + HPX whilst avoid
// any CPU/GPU barriers
//=================================================================================================

// Define types: A lot of this can be done automatically, however, here we want to show the manual
// approach (as using different types/ifdefs can allow us to specialize kernels for specific hardware
// if required. 
//
using float_t = float;
// Use correct device exeuction space and memory spaces depending on the activated device 
// execution space
#ifdef KOKKOS_ENABLE_CUDA
// Pick executor type
using device_executor_t = hpx::kokkos::cuda_executor;
// Define Kokkos View types to be used! Must be using MemoryUnmanaged to allow memory recycling
using kokkos_device_view_t = Kokkos::View<float_t*, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>; 
using kokkos_host_view_t = Kokkos::View<float_t*, Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>; 
// Define CPPuddle recycling allocators to be used with the views
using device_allocator_t = cppuddle::memory_recycling::recycle_allocator_cuda_device<float_t>;
using host_allocator_t = cppuddle::memory_recycling::recycle_allocator_cuda_host<float_t>;
#elif KOKKOS_ENABLE_HIP
// Pick executor type
using device_executor_t = hpx::kokkos::hip_executor;
// Define Kokkos View types to be used! Must be using MemoryUnmanaged to allow memory recycling
using kokkos_device_view_t = Kokkos::View<float_t*, Kokkos::HIPSpace, Kokkos::MemoryUnmanaged>; 
using kokkos_host_view_t = Kokkos::View<float_t*, Kokkos::HIPHostPinnedSpace, Kokkos::MemoryUnmanaged>; 
// Define CPPuddle recycling allocators to be used with the views
using device_allocator_t = cppuddle::memory_recycling::recycle_allocator_hip_device<float_t>;
using host_allocator_t = cppuddle::memory_recycling::recycle_allocator_hip_host<float_t>;
#elif KOKKOS_ENABLE_SYCL
// Pick executor type
using device_executor_t = hpx::kokkos::sycl_executor;
// Define Kokkos View types to be used! Must be using MemoryUnmanaged to allow memory recycling
using kokkos_device_view_t = Kokkos::View<float_t*, Kokkos::SYCLDeviceUSMSpace, Kokkos::MemoryUnmanaged>; 
using kokkos_host_view_t = Kokkos::View<float_t*, Kokkos::SYCLDeviceHostSpace, Kokkos::MemoryUnmanaged>; 
// Define CPPuddle recycling allocators to be used with the views
using device_allocator_t = cppuddle::memory_recycling::recycle_allocator_sycl_device<float_t>;
using host_allocator_t = cppuddle::memory_recycling::recycle_allocator_sycl_host<float_t>;
#else
#error "Example assumes both a host and a device Kokkos execution space are available"
#endif
// Plug together the defined Kokkos views with the recycling CPPuddle allocators
// This yields a new type that can be used just like a normal Kokkos View but gets its memory from 
// CPPuddle.
using recycling_device_view_t =
    cppuddle::memory_recycling::recycling_view<kokkos_device_view_t,
                                               device_allocator_t, float_t>;
using recycling_host_view_t =
    cppuddle::memory_recycling::recycling_view<kokkos_host_view_t,
                                               host_allocator_t, float_t>;

// Run host kernels on HPX execution space:
using host_executor_t = hpx::kokkos::hpx_executor;
// Serial executor can actually work well, too when interleaving multiple Kokkos kernels to
// achieve multicore usage. However, be aware that this only works for Kokkos kernels that are not
// using team policies / scratch memory (those use a global barrier across all Serial execution 
// spaces):
// using host_executor_t = hpx::kokkos::serial_executor;

// The actual compute kernel: Simply defines a exeuction policy with the given executor and runs the
// kernel with a Kokkos::parallel_for
template <typename executor_t, typename view_t>
void kernel_add(executor_t &executor, const size_t entries_per_task,
                const view_t &input_a, const view_t &input_b, view_t &output_c)
{
  // Define exeuction policy
  auto execution_policy = Kokkos::Experimental::require(
      Kokkos::RangePolicy<decltype(executor.instance())>(
          executor.instance(), 0, entries_per_task),
      Kokkos::Experimental::WorkItemProperty::HintLightWeight);

  // Run Kernel with execution policy (and give it some name ideally)
  Kokkos::parallel_for(
      "sample vector add kernel", execution_policy,
      KOKKOS_LAMBDA(size_t index) {
        output_c[index] = input_a[index] + input_b[index];
      });
}

/** Method that demonstrates how one might launch a Kokkos kernel with HPX and
 * CPPuddle recycled memory/executors! By using CPPuddle allocators to avoid
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
  // 1. Create recycled Kokkos host views
  recycling_host_view_t host_a(entries_per_task);
  recycling_host_view_t host_b(entries_per_task);
  recycling_host_view_t host_c(entries_per_task);

  // 2. Host-side preprocessing (usually: communication, here fill dummy input)
  for (size_t i = 0; i < entries_per_task; i++) {
    host_a[i] = 1.0;
    host_b[i] = 2.0;
  }

  // 3. Check GPU utilization - Method will return true if there is an executor
  // in the pool that does currently not exceed its queue limit (tracked by
  // RAII, no CUDA/HIP/SYCL API calls involved)
  bool device_executor_available =
      cppuddle::executor_recycling::executor_pool::interface_available<
          device_executor_t, cppuddle::executor_recycling::
                                 round_robin_pool_impl<device_executor_t>>(
          max_queue_length, gpu_id);

  // 4. Run Kernel on either CPU or GPU
  if (!device_executor_available) {
    // 4a. Launch CPU Fallback  Version
    number_cpu_kernel_launches++;
    // Draw host executor
    cppuddle::executor_recycling::executor_interface<
        host_executor_t,
        cppuddle::executor_recycling::round_robin_pool_impl<host_executor_t>>
        executor(gpu_id); // Wrapper that draws executor from the pool

    // Launch
    kernel_add(static_cast<host_executor_t&>(executor), entries_per_task, host_a, host_b, host_c);
    
    // Sync kernel
    static_cast<host_executor_t&>(executor).instance().fence();

  } else {
    number_gpu_kernel_launches++;
    // 4b. Create per_task device-side views (using recylced device memory)
    // and draw GPU executor from CPPuddle executor pool
    // Draw host device
    cppuddle::executor_recycling::executor_interface<
        device_executor_t,
        cppuddle::executor_recycling::round_robin_pool_impl<device_executor_t>>
        executor(gpu_id); // Wrapper that draws executor from the pool

    recycling_device_view_t device_a(entries_per_task);
    recycling_device_view_t device_b(entries_per_task);
    recycling_device_view_t device_c(entries_per_task);

    // 4c. Launch data transfers and kernel
    Kokkos::deep_copy(executor.interface.instance(), device_a, host_a);
    Kokkos::deep_copy(executor.interface.instance(), device_b, host_b);

    kernel_add(static_cast<device_executor_t &>(executor), entries_per_task,
               device_a, device_b, device_c);

    auto transfer_fut = hpx::kokkos::deep_copy_async(
        executor.interface.instance(), host_c, device_c);
    transfer_fut.get();

    // 5. Host-side postprocessing (usually: communication, here: check
    // correctness)
    for (size_t i = 0; i < entries_per_task; i++) {
      if (host_c[i] != 1.0 + 2.0) {
        std::cerr << "Task " << task_id << " contained wrong results!!"
                  << std::endl;
        break;
      }
    }
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

/** HPX uses either callbacks or event polling to implement its HPX-Kokkos futures.
 * Polling usually has the superior performance, however, it requires that the
 * polling is initialized at startup (or at least before the HPX-Kokkos futures are
 * used). The CPPuddle executor pool also needs initialzing as we need to set it
 * to a specified number of executors (which CPPuddle cannot know without the
 * number_gpu_executors parameter). We will use the round_robin_pool_impl for
 * simplicity. A priority_pool_impl is also available.
 */
void init_executor_pool_and_polling(const size_t number_gpu_executors,
                                    const size_t number_cpu_executors,
                                    const size_t gpu_id) {
  assert(gpu_id == 0); // MultiGPU not used in this example
  // Init polling
  hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
  // Init device executors
  cppuddle::executor_recycling::executor_pool::init_executor_pool<
      device_executor_t,
      cppuddle::executor_recycling::round_robin_pool_impl<device_executor_t>>(
      gpu_id, number_gpu_executors, hpx::kokkos::execution_space_mode::independent);
  /* // Init host executors (fixed to 256) */
  cppuddle::executor_recycling::executor_pool::init_all_executor_pools<
      host_executor_t,
      cppuddle::executor_recycling::round_robin_pool_impl<host_executor_t>>(
      number_cpu_executors, hpx::kokkos::execution_space_mode::independent);
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
  // Init Kokkos 
  Kokkos::initialize();
  // Init/Finalize Kokkos alternative using RAII:
  /* hpx::kokkos::ScopeGuard g(argc, argv); */
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
  size_t number_cpu_executors = 128;
  size_t gpu_id = 0;
  if(!process_cli_options(argc, argv, entries_per_task, number_tasks,
                      in_order_repetitions, number_repetitions,
                      number_gpu_executors, max_queue_length)) {
    return hpx::finalize(); // problem with CLI parameters detected -> exiting..
  }

  // Init HPX CUDA polling + executor pool
  hpx::cout << "Start initializing CUDA polling and executor pool..." << std::endl;
  init_executor_pool_and_polling(number_gpu_executors, number_cpu_executors,
                                 gpu_id);
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
  hpx::cuda::experimental::detail::unregister_polling(
      hpx::resource::get_thread_pool(0));

  // Cleanup (executor_pool cleanup required to deallocate all Kokkos execution
  // spaces before Kokkos finalize is called)
  cppuddle::executor_recycling::executor_pool::cleanup<
      device_executor_t,
      cppuddle::executor_recycling::round_robin_pool_impl<device_executor_t>>();
  cppuddle::executor_recycling::executor_pool::cleanup<
      host_executor_t,
      cppuddle::executor_recycling::round_robin_pool_impl<host_executor_t>>();
  cppuddle::memory_recycling::finalize();
  Kokkos::finalize(); // only required if hpx-kokkos Scope Guard is not used
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
