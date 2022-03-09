// Copyright (c) 2022-2022 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <hpx/futures/future.hpp>
#undef NDEBUG


#include "../include/aggregation_manager.hpp"
#include "../include/cuda_buffer_util.hpp"

#include <boost/program_options.hpp>



//===============================================================================
//===============================================================================
// Stream benchmark

size_t launch_counter = 0;
template <typename float_t>
void triad_kernel(float_t *A, const float_t *B, const float_t *C, const float_t scalar, const size_t start_id, const size_t kernel_size, const size_t problem_size) {
  for (auto int i = 0; i < kernel_size && i + start_id < problem_size; i++) {
    A[i] = B[i] + scalar * C[i];
  }
  launch_counter++;
}


//===============================================================================
//===============================================================================
// TODO Add to shared headerfile (aggregation_test_util.hpp?)...
//
/// Dummy CPU executor (providing correct interface but running everything
/// immediately Intended for testing the aggregation on the CPU, not for
/// production use!
struct Dummy_Executor {
  /// Executor is always ready
  hpx::lcos::future<void> get_future() {
    // To trigger interruption in exeuctor coalesing manually with the promise
    // For a proper CUDA executor we would get a future that's ready once the
    // stream is ready of course!
    return hpx::make_ready_future();
  }
  /// post -- executes immediately
  template <typename F, typename... Ts> void post(F &&f, Ts &&...ts) {
    f(std::forward<Ts>(ts)...);
  }
  /// async -- executores immediately and returns ready future
  template <typename F, typename... Ts>
  hpx::lcos::future<void> async(F &&f, Ts &&...ts) {
    f(std::forward<Ts>(ts)...);
    return hpx::make_ready_future();
  }
};


//===============================================================================
//===============================================================================
int hpx_main(int argc, char *argv[]) {
  // Init parameters
  std::string filename{};
  size_t problem_size{0};
  size_t kernel_size{0};
  size_t max_slices{0};
  size_t repetitions{0};
  size_t number_aggregation_executors{0};
  size_t number_underlying_executors{0};
  bool print_launch_counter{false};
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
                         "Print number of kernel launches as verification the aggregation works")("outputfile",
                                                      boost::program_options::
                                                          value<std::string>(
                                                              &filename)
                                                              ->default_value(
                                                                  ""),
                                                      "Redirect stdout/stderr "
                                                      "to this file");

      boost::program_options::variables_map vm;
      boost::program_options::parsed_options options =
          parse_command_line(argc, argv, desc);
      boost::program_options::store(options, vm);
      boost::program_options::notify(vm);

      if (vm.count("help") == 0u) {
        /*hpx::cout << "Running with parameters:" << std::endl
                  << "--problem_size=" << problem_size << std::endl
                  << "--kernel_size=" << kernel_size << std::endl
                  << "--max_slices=" << max_slices << std::endl
                  << "--number_aggregation_executors=" << number_aggregation_executors << std::endl
                  << "--number_underlying_executors=" << number_underlying_executors << std::endl
                  << "--repetitions=" << repetitions << std::endl
                  << "--print_launch_counter=" << print_launch_counter << std::endl
                  << "--outputfile=" << filename << std::endl;*/
      } else {
        hpx::cout << desc << std::endl;
        return hpx::finalize();
      }
    } catch (const boost::program_options::error &ex) {
      hpx::cout << "CLI argument problem found: " << ex.what() << '\n';
    }
    if (!filename.empty()) {
      freopen(filename.c_str(), "w", stdout); // NOLINT
      freopen(filename.c_str(), "w", stderr); // NOLINT
    }
  }

  stream_pool::init<Dummy_Executor, round_robin_pool<Dummy_Executor>>(number_underlying_executors);
  static const char kernelname[] = "cpu_triad";
  using executor_pool = aggregation_pool<kernelname, Dummy_Executor,
                                        round_robin_pool<Dummy_Executor>>;
  //executor_pool::init(number_aggregation_executors, max_slices, Aggregated_Executor_Modes::EAGER);
  executor_pool::init(number_aggregation_executors, max_slices, Aggregated_Executor_Modes::STRICT);
  /* std::cerr << "created pool with slices: " << max_slices << std::endl; */

  using float_t = float;

  for (size_t repetition = 0; repetition < repetitions; repetition++) {
    std::cerr << hpx::get_worker_thread_num() << "Starting repetition: " << repetition << std::endl;

    std::vector<float_t> A(problem_size, 0.0);
    std::vector<float_t> B(problem_size, 2.0);
    std::vector<float_t> C(problem_size, 1.0);
    const float_t scalar = 3.0;

    size_t number_tasks = problem_size / kernel_size;
    std::vector<hpx::lcos::future<void>> futs;

    for (size_t task_id = 0; task_id < number_tasks; task_id++) {
      // Concurrency Wrapper: Splits stream benchmark into #number_tasks tasks
      /* std::cerr << hpx::get_worker_thread_num() << "Scheduling task: " << task_id << std::endl; */
      futs.push_back(hpx::async([&, task_id]() {
        /* std::cerr<< hpx::get_worker_thread_num() << "Running task: " << task_id << std::endl; */
        auto slice_fut1 = executor_pool::request_executor_slice();
        if (slice_fut1.has_value()) {
          // Work aggregation Wrapper: Recombines (some) tasks, depending on the
          // number of slices
        hpx::lcos::future<void> current_fut = slice_fut1.value().then([&, task_id](auto &&fut) {
            /* std::cerr << hpx::get_worker_thread_num() */
            /* << " Running kernel continuation: " << task_id << std::endl; */
            auto slice_exec = fut.get();
            //std::cerr << task_id << " ";

            /* auto alloc = */
            /*     slice_exec.template make_allocator<float_t, */
            /*                                        std::allocator<float_t>>(); */
            /* // Start the actual task */
            /* std::vector<float_t, decltype(alloc)> local_A( */
            /*     slice_exec.number_slices * kernel_size, float_t{}, alloc); */
            /* std::vector<float_t, decltype(alloc)> local_B( */
            /*     slice_exec.number_slices * kernel_size, float_t{}, alloc); */
            /* std::vector<float_t, decltype(alloc)> local_C( */
            /*     slice_exec.number_slices * kernel_size, float_t{}, alloc); */
            /* for (size_t i = task_id * kernel_size, j = 0; */
            /*      i < problem_size && j < kernel_size; i++, j++) { */
            /*   local_B[slice_exec.id * kernel_size + j] = B[i]; */
            /*   local_C[slice_exec.id * kernel_size + j] = C[i]; */
            /*   local_A[slice_exec.id * kernel_size + j] = 0.0; */
            /* } */
            /* const size_t start_id = */
            /*     task_id * kernel_size - slice_exec.id * kernel_size; */
            /* auto kernel_done = slice_exec.async( */
            /*     triad_kernel<float_t>, local_A.data(), local_B.data(), */
            /*     local_C.data(), scalar, 0, */
            /*     kernel_size * slice_exec.number_slices, problem_size); */
            /* kernel_done.get(); */
            /* for (size_t i = task_id * kernel_size, j = 0; */
            /*      i < problem_size && j < kernel_size; i++, j++) { */
            /*   A[i] = local_A[slice_exec.id * kernel_size + j]; */
            /* } */
            // end actual task
            /* std::cerr << hpx::get_worker_thread_num() */
            /* << " End of task: " << task_id << std::endl; */
          }); 
        //current_fut.get();
        return current_fut;
        } else {
          std::cerr << "Executor was not properly initialized!" << std::endl;
          return hpx::lcos::make_ready_future();
        }
      })); 
      /* std::cerr<< hpx::get_worker_thread_num() << "Waiting ... " << task_id << std::endl; */
      /* futs[futs.size()-1].get(); */
      /* std::cerr<< hpx::get_worker_thread_num() << "Waiting done " << task_id << std::endl; */
    }
    auto final_fut = hpx::lcos::when_all(futs);
    final_fut.get();

    /* for (size_t i = 0; i < problem_size; i++) { */
    /*   if (i < 100) */
    /*     hpx::cout << A[i] << " "; */
    /* } */
  }
  hpx::cout << std::endl;
  hpx::cout << "Kernel launch counter: " << launch_counter << std::endl;

  // Flush outout and wait a second for the (non hpx::cout) output to have it in the correct
  // order for the ctests
  //std::flush(hpx::cout);
  //sleep(1);

  recycler::force_cleanup(); // Cleanup all buffers and the managers
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
