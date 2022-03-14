// Copyright (c) 2022-2022 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <chrono>
#include <hpx/futures/future.hpp>
//#undef NDEBUG


#include "../include/aggregation_manager.hpp"
#include "../include/cuda_buffer_util.hpp"

#include <boost/program_options.hpp>



//===============================================================================
//===============================================================================
// Stream benchmark

template <typename float_t>
__global__ void __launch_bounds__(1024, 4) triad_kernel(float_t *A, const float_t *B, const float_t *C, const float_t scalar, const size_t start_id, const size_t kernel_size, const size_t problem_size) {
  const size_t i = start_id + blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = B[i] + scalar * C[i];
}


//===============================================================================
//===============================================================================
int hpx_main(int argc, char *argv[]) {
  // Init parameters
  size_t problem_size{0};
  size_t kernel_size{0};
  size_t max_slices{0};
  size_t repetitions{0};
  size_t number_aggregation_executors{0};
  size_t number_underlying_executors{0};
  bool print_launch_counter{false};
  std::string executor_type_string{};
  Aggregated_Executor_Modes executor_mode{Aggregated_Executor_Modes::EAGER};
  std::string filename{};
  {
    try {
      boost::program_options::options_description desc{"Options"};
      desc.add_options()(
          "help",
          "Help screen")("problem_size",
           boost::program_options::value<size_t>(&problem_size)
           ->default_value(12800),
           "Number of vector elements for triad test")("kernel_size",
           boost::program_options::value<size_t>(&kernel_size)
           ->default_value(128),
           "Number of vector elements per kernel launch")("max_slices",
           boost::program_options::value<size_t>(&max_slices)
           ->default_value(8),
           "Max number of work aggregation slices")("number_aggregation_executors",
           boost::program_options::value<size_t>(&number_aggregation_executors)
           ->default_value(8),
           "Start number of aggregation executors")("number_underlying_executors",
           boost::program_options::value<size_t>(&number_underlying_executors)
           ->default_value(8),
           "Number of host executors that are used")("repetitions",
           boost::program_options::value<size_t>(&repetitions)
           ->default_value(1),
           "Number of times the test should be repeated")("print_launch_counter",
           boost::program_options::value<bool>(&print_launch_counter)
           ->default_value(true),
           "Print number of kernel launches")("executor_type",
           boost::program_options::value<std::string>(&executor_type_string)
           ->default_value("EAGER"),
           "Aggregation executor type [EAGER,STRICT,ENDLESS")("outputfile",
           boost::program_options::value<std::string>(&filename)->default_value(""),
           "Redirect stdout/stderr to this file");

      boost::program_options::variables_map vm;
      boost::program_options::parsed_options options =
          parse_command_line(argc, argv, desc);
      boost::program_options::store(options, vm);
      boost::program_options::notify(vm);

      if (vm.count("help") == 0u) {
        hpx::cout << "Running with parameters:" << std::endl
                  << "--problem_size=" << problem_size << std::endl
                  << "--kernel_size=" << kernel_size << std::endl
                  << "--max_slices=" << max_slices << std::endl
                  << "--number_aggregation_executors="
                  << number_aggregation_executors << std::endl
                  << "--number_underlying_executors="
                  << number_underlying_executors << std::endl
                  << "--repetitions=" << repetitions << std::endl
                  << "--print_launch_counter=" << print_launch_counter
                  << std::endl
                  << "--executor_type=" << executor_type_string << std::endl
                  << "--outputfile=" << filename << std::endl;
      } else {
        hpx::cout << desc << std::endl;
        return hpx::finalize();
      }
      if (executor_type_string == "EAGER") {
        executor_mode = Aggregated_Executor_Modes::EAGER;
      } else if (executor_type_string == "STRICT") {
        executor_mode = Aggregated_Executor_Modes::STRICT;
      } else if (executor_type_string == "ENDLESS") {
        executor_mode = Aggregated_Executor_Modes::ENDLESS;
      } else {
        std::cerr << "ERROR: Unknown executor mode " << executor_type_string
                  << "\n Valid choices are: EAGER,STRICT,ENDLESS" << std::endl;
        exit(1);
      }
    } catch (const boost::program_options::error &ex) {
      hpx::cout << "CLI argument problem found: " << ex.what() << '\n';
    }
    if (!filename.empty()) {
      freopen(filename.c_str(), "w", stdout); // NOLINT
      freopen(filename.c_str(), "w", stderr); // NOLINT
    }
  }

  hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));

  using executor_t = hpx::cuda::experimental::cuda_executor;
  stream_pool::init<executor_t, round_robin_pool<executor_t>>(
      number_underlying_executors, 0, true);
  static const char kernelname2[] = "cuda_triad";
  using executor_pool = aggregation_pool<kernelname2, executor_t,
                                         round_robin_pool<executor_t>>;
  executor_pool::init(number_aggregation_executors, max_slices, executor_mode);

  using float_t = float;
  //epsilon for comparison
  double epsilon;
  if (sizeof(float_t) == 4) {
    epsilon = 1.e-6;
  } else if (sizeof(float_t) == 8) {
    epsilon = 1.e-13;
  } else {
    hpx::cout << "Unexpected float size " << sizeof(float_t) << std::endl;
    hpx::cout << "Use either double or float - falling back to float epsilon..."
              << std::endl;
    epsilon = 1.e-6;
  }
  const size_t variant = 0;

  if (variant == 0) {
    for (size_t repetition = 0; repetition < repetitions; repetition++) {

      std::vector<float_t> A(problem_size, 0.0);
      std::vector<float_t> B(problem_size, 2.0);
      std::vector<float_t> C(problem_size, 1.0);
      float_t scalar = 3.0;

      size_t number_tasks = problem_size / kernel_size;
      std::vector<hpx::lcos::future<void>> futs;
      cudaError_t(*func)(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t) = cudaMemcpyAsync;


      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();
      for (size_t task_id = 0; task_id < number_tasks; task_id++) {
        // Concurrency Wrapper: Splits stream benchmark into #number_tasks tasks
        futs.push_back(hpx::async([&, task_id]() {
          auto slice_fut1 = executor_pool::request_executor_slice();
          if (slice_fut1.has_value()) {
            // Work aggregation Wrapper: Recombines (some) tasks, depending on the
            // number of slices
            hpx::lcos::future<void> current_fut =
                slice_fut1.value().then([&, task_id](auto &&fut) {
                  auto slice_exec = fut.get();

                  auto alloc_host = slice_exec.template make_allocator<
                      float_t, recycler::detail::cuda_pinned_allocator<float_t>>();
                  auto alloc_device = slice_exec.template make_allocator<
                      float_t, recycler::detail::cuda_device_allocator<float_t>>();

                  // Start the actual task

                  // todo -- one slice gets a buffer that's not vaild anymore
                  std::vector<float_t, decltype(alloc_host)> local_A(
                      slice_exec.number_slices * kernel_size, float_t{}, alloc_host);

                  recycler::cuda_aggregated_device_buffer<float_t,
                                                          decltype(alloc_device)>
                      device_A(slice_exec.number_slices * kernel_size, 0,
                               alloc_device);

                  std::vector<float_t, decltype(alloc_host)> local_B(
                      slice_exec.number_slices * kernel_size, float_t{},
                      alloc_host);
                  recycler::cuda_aggregated_device_buffer<float_t,
                                                          decltype(alloc_device)>
                      device_B(slice_exec.number_slices * kernel_size, 0,
                               alloc_device);
                  
                  std::vector<float_t, decltype(alloc_host)> local_C(
                      slice_exec.number_slices * kernel_size, float_t{},
                      alloc_host);
                  recycler::cuda_aggregated_device_buffer<float_t,
                                                          decltype(alloc_device)>
                      device_C(slice_exec.number_slices * kernel_size, 0,
                               alloc_device);

                  for (size_t i = task_id * kernel_size, j = 0;
                       i < problem_size && j < kernel_size; i++, j++) {
                    local_B[slice_exec.id * kernel_size + j] = B[i];
                    local_C[slice_exec.id * kernel_size + j] = C[i];
                    local_A[slice_exec.id * kernel_size + j] = 0.0;
                  }
                  slice_exec.template post<decltype(func)>(
                      cudaMemcpyAsync, device_B.device_side_buffer,
                      local_B.data(),
                      slice_exec.number_slices * kernel_size * sizeof(float_t),
                      cudaMemcpyHostToDevice);
                  slice_exec.template post<decltype(func)>(
                      cudaMemcpyAsync, device_C.device_side_buffer,
                      local_C.data(),
                      slice_exec.number_slices * kernel_size * sizeof(float_t),
                      cudaMemcpyHostToDevice);
                  const size_t start_id =
                      task_id * kernel_size - slice_exec.id * kernel_size;

                  dim3 const grid_spec(slice_exec.number_slices, 1, 1);
                  dim3 const threads_per_block(kernel_size, 1, 1);
                  size_t arg1 = 0;
                  size_t arg2 = kernel_size * slice_exec.number_slices;
                  void *args[] = {&(device_A.device_side_buffer),
                                  &(device_B.device_side_buffer),
                                  &(device_C.device_side_buffer),
                                  &scalar,
                                  &arg1,
                                  &arg2,
                                  &problem_size};
                  slice_exec.post(
                      cudaLaunchKernel<decltype(triad_kernel<float_t>)>,
                      triad_kernel<float_t>, grid_spec, threads_per_block, args,
                      0);

                  auto result_fut = slice_exec.template async<decltype(func)>(
                      cudaMemcpyAsync, local_A.data(), device_A.device_side_buffer,
                      slice_exec.number_slices * kernel_size * sizeof(float_t),
                      cudaMemcpyDeviceToHost);
                  result_fut.get();
                  for (size_t i = task_id * kernel_size, j = 0;
                       i < problem_size && j < kernel_size; i++, j++) {
                    A[i] = local_A[slice_exec.id * kernel_size + j];
                  }
                // end actual task
              });
            // current_fut.get();
            return current_fut;
          } else {
            hpx::cout << "ERROR: Executor was not properly initialized!" << std::endl;
            return hpx::lcos::make_ready_future();
          }
        })); 
      }
      auto final_fut = hpx::lcos::when_all(futs);
      final_fut.get();
      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      /* std::cin.get(); */

      bool results_correct = true;
      for (size_t i = 0; i < problem_size; i++) {
        /* if (i < 100) */
        /*   hpx::cout << A[i] << " "; */

        // result should be 5.0
        if (std::abs(A[i] - 5.0) > epsilon) {
          hpx::cout << "Found error at " << i << " : " << A[i]
                    << " instead of 5.0" << std::endl;
          assert(false); // in debug build: crash
          results_correct = false;
          break;
        }
      }
      if (!results_correct) {
        hpx::cout << "ERROR in repetition " << repetition << ": Wrong results"
                  << std::endl;
      } else {
        hpx::cout << "SUCCESS: Repetition " << repetition << " runtime = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                           begin)
                         .count()
                  << "[µs]" << std::endl;
      }
    }
    hpx::cout << std::endl;
  } else if (variant == 1) {
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::vector<float_t> A(problem_size, 0.0);
    std::vector<float_t> B(problem_size, 2.0);
    std::vector<float_t> C(problem_size, 1.0);
    recycler::cuda_device_buffer<float_t> device_A(problem_size, 0);
    recycler::cuda_device_buffer<float_t> device_B(problem_size, 0);
    recycler::cuda_device_buffer<float_t> device_C(problem_size, 0);
    cudaMemcpy(device_A.device_side_buffer, A.data(),
               problem_size * sizeof(float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B.device_side_buffer, B.data(),
               problem_size * sizeof(float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_C.device_side_buffer, C.data(),
               problem_size * sizeof(float_t), cudaMemcpyHostToDevice);
    float_t scalar = 3.0;
    dim3 const grid_spec(problem_size / kernel_size, 1, 1);
    dim3 const threads_per_block(kernel_size, 1, 1);
    size_t arg1 = 0;
    size_t arg2 = 0;

    triad_kernel<float_t><<<grid_spec, threads_per_block>>>(
        (device_A.device_side_buffer), (device_B.device_side_buffer),
        (device_C.device_side_buffer), scalar, arg1, arg2, problem_size);
    
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "Orig runtime = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;
    for (size_t repetition = 0; repetition < repetitions; repetition++) {

      size_t number_tasks = problem_size / kernel_size;
      std::vector<hpx::lcos::future<void>> futs;
      cudaError_t(*func)(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t) = cudaMemcpyAsync;


      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();
      for (size_t task_id = 0; task_id < number_tasks; task_id++) {
        // Concurrency Wrapper: Splits stream benchmark into #number_tasks tasks
        futs.push_back(hpx::async([&, task_id]() {
          auto slice_fut1 = executor_pool::request_executor_slice();
          if (slice_fut1.has_value()) {
            // Work aggregation Wrapper: Recombines (some) tasks, depending on the
            // number of slices
            hpx::lcos::future<void> current_fut =
                slice_fut1.value().then([&, task_id](auto &&fut) {
                  auto slice_exec = fut.get();

                  dim3 const grid_spec(slice_exec.number_slices, 1, 1);
                  dim3 const threads_per_block(kernel_size, 1, 1);
                  size_t arg1 = 0;
                  size_t arg2 = kernel_size * slice_exec.number_slices;
                  size_t start_id =
                      task_id * kernel_size - slice_exec.id * kernel_size;
                  void *args[] = {&(device_A.device_side_buffer),
                                  &(device_B.device_side_buffer),
                                  &(device_C.device_side_buffer),
                                  &scalar,
                                  &start_id,
                                  &arg2,
                                  &problem_size};
                  auto result_fut = slice_exec.async(
                      cudaLaunchKernel<decltype(triad_kernel<float_t>)>,
                      triad_kernel<float_t>, grid_spec, threads_per_block, args,
                      0);

                  result_fut.get();
                // end actual task
              });
            // current_fut.get();
            return current_fut;
            return hpx::make_ready_future();
          } else {
            hpx::cout << "ERROR: Executor was not properly initialized!" << std::endl;
            return hpx::lcos::make_ready_future();
          }
        })); 
      }
      auto final_fut = hpx::lcos::when_all(futs);
      final_fut.get();
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "SUCCESS: Repetition " << repetition << " runtime = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;
    }
    cudaMemcpy(device_C.device_side_buffer, C.data(),
               problem_size * sizeof(float_t), cudaMemcpyHostToDevice);
    hpx::cout << std::endl;
  }

  // Flush outout and wait a second for the (non hpx::cout) output to have it in the correct
  // order for the ctests
  /* std::flush(hpx::cout); */
  /* sleep(1); */

  hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
  recycler::force_cleanup(); // Cleanup all buffers and the managers
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
