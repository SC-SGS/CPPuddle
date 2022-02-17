// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <atomic>
#include <chrono>
#include <cstdio>
#include <hpx/collectives/latch.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <iostream>
#include <mutex>
#include <string>
#include <tuple>
#include <typeinfo>

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/promise.hpp>

#include <boost/program_options.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>

#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"
#include "../include/stream_manager.hpp"

// TODO Add work aggregation class
// Needs
// - Mutex/flag to stop giving out new slices?
// - A slice counter
// - The actual executor we are going to use
// - A yield buffer(slice?) method

class work_aggregator {
private:
  // Protect with mutex
  const unsigned int max_number_slices{4};
  unsigned int used_number_slices;
  bool execution_phase; // stop giving out slices once this is done
  // Each sort of needs a latch from the interface
  void launch_aggregated_kernel();
  // Each sort of needs a latch from the interface
  void move_aggregated_slice_to_device();
  // Said mutex
  std::mutex mut;
  // Continuation
  // Decrase function objects latches by 4-X (X being the number of used slices
  // so far, but at least 1)
  void stream_notify_ready();

public:
  // get created / pulled from stream manager with the creation of the first
  // interface stream is pulled from stream_manager
  work_aggregator(void);
  // Should get destroyed/reset once all slices are given up
  ~work_aggregator(void);

  // Slice operation
  void yield_slice();
  void move_slice_to_device();
  void move_slice_from_device();
  void set_slice();
  void launch_kernel();
  // allocator;
};

// Have "function objects" with latches attached to the work aggegator
// -----------------------------------------------------------------
// Have continuation trigger the latches until either: the thing launches OR we
// are at the first slice and it's not yet filled Once the launches start, no
// more slices are given out, yet existing slices can launch more stuff (again,
// being triggered by subsqueent continuations) Once all slices (local RAII
// interfaces) are destroyed, the work_aggregator is destroyed (the stream
// persists though) Interestingly, the actual function call looks the same on
// all participating tasks - assuming I can get the buffers cast
//
// Stream (from stream_manager)
// ->
// Work aggregator (tmp) --> (create -> attach phase -> running phase ->
// destroyed once all slices are gone)
// ->
// Work slice (RAII) -- Function call object (latch)
// ->
//  -
// ->
// futures

// open phase (give away new slices)
// close phase (first function being launched: stop giving away new slices

// Keep track of phases (open-phase, closed-phase)
// Give away slices (work aggregator interfaces)
// Have a stream
// Construct on the fly, be able to have multiple ones (though only one in the
// open phase at a time)
class work_aggregator_bla {};

// Provide RAII
// Provide function calls
class work_aggregator_interface {};

void print_stuff1(int used_slices, int i) {
  std::cout << "i is " << i << "(Slice " << used_slices << ")" << std::endl;
}
void print_stuff2(int used_slices, int i, double d) {
  std::cout << "i is " << i << std::endl;
  std::cout << "d is " << d << std::endl;
  std::cout << "(Slice is " << used_slices << ")" << std::endl;
}
void print_stuff3(const std::atomic<int> &used_slices, int i) {
  const int bla = used_slices;
  std::cout << "i is " << i << "(Slice " << bla << ")" << std::endl;
}
/*void print_stuff1(int *used_slices, int i) {
  std::cout << "i is " << i << "(Slice " << *used_slices << ")" << std::endl;
}
void print_stuff2(int *used_slices, int i, double d) {
  std::cout << "i is " << i << std::endl;
  std::cout << "d is " << d << std::endl;
  std::cout << "(Slice is " << *used_slices << ")" << std::endl;
}*/

// 3 launch conditions:
// 1. previous functino call on the "aggegated executor" has been laucnhed (see
// current_future)
// 2. No more slices are given away
// --> 2.1 Either stream has become ready OR
// --> 2.2 we have given away enough slices
// 3. All slices that have been given away have visited the function call
// (slices_ready promise should be triggered)

class function_call_aggregator {
public:
  // I would sort of need that for every function call?
  // getting triggered once stuff is ready
  hpx::lcos::local::promise<void> launch_promise;
  hpx::lcos::future<void> current_future = launch_promise.get_future();
  hpx::lcos::local::promise<void> slices_ready_promise;
  hpx::lcos::future<void> all_slices_ready = slices_ready_promise.get_future();

  const int max_slices = 4;
  int current_slice_number = 0;

public:
  // Just launch it as the continuation
  // OR: use these as actual function objects
  template <typename F, typename... Ts> void post(F &&f, Ts &&...ts) {
    int &current_slice_number = this->current_slice_number;
    current_future = hpx::lcos::when_all(current_future, all_slices_ready)
                         .then([=, &current_slice_number](auto &&old_fut) {
                           std::cout << "starting function continuation"
                                     << std::endl;
                           f(current_slice_number, ts...);
                           return;
                         });
    current_slice_number++;
  }

  // Using a tuple as args (actually easy to store this stuff this way)
  // struggles with reference though (have to use pointers instead?)
  /*template <typename F, typename... Ts> void post2(F &&f, Ts &&...ts) {
    auto args = std::make_tuple(&current_slice_number, std::forward<Ts>(ts)...);
    // Check if we're using the same stuff
    if (f==f && args == args) std::cout << "yay" << std::endl;
    current_future = hpx::lcos::when_all(current_future, all_slices_ready).then(
        [f = std::move(f), args = std::move(args)](auto &&old_fut) {
          std::cout << "starting function continuation" << std::endl;
          std::apply(f, std::move(args));
          return;
        });
    this->current_slice_number++;
  }*/
};

// slice 1 to be the master slice launching this stuff
// each one has a slice id
template <const char *kernelname, typename Executor>
class aggregated_function_call {
private:
  std::atomic<int> slice_counter = 0; // not necessasry
  hpx::lcos::local::promise<void> slices_ready_promise;
  hpx::lcos::future<void> all_slices_ready = slices_ready_promise.get_future();
  const size_t number_slices;

public:
  // TODO Add constructor
  aggregated_function_call(const size_t number_slices)
      : number_slices(number_slices) {}
  template <typename F, typename... Ts>
  void post_when(hpx::lcos::future<void> &stream_future, F &&f, Ts &&...ts) {
    slice_counter++;
    std::atomic<int> &current_slice_counter = this->slice_counter;
    std::cout << "Slices counter ... " << slice_counter << std::endl;

    if (slice_counter == 1) {
      stream_future = hpx::lcos::when_all(stream_future, all_slices_ready)
                          .then([=, &current_slice_counter](auto &&old_fut) {
                            std::cout << "starting function_call continuation"
                                      << std::endl;
                            // TODO modify according to slices (launch either X
                            // seperate ones, or have one specialization
                            // launching stuff...)
                            f(current_slice_counter, ts...);
                            std::cout << kernelname << std::endl;
                            ;
                            return;
                          });
    }
    if (slice_counter == number_slices) {
      std::cout << "Starting slices ready..." << std::endl;
      slices_ready_promise.set_value();
    }
  }
  // TODO async call
};

//===============================================================================
//===============================================================================
/// Executor Class that aggregates function calls for specific kernels
/** Executor is not meant to be used directly. Instead it yields multiple
 * Executor_Slice objects. These serve as interfaces. Slices from the same
 * Aggregated_Executor are meant to execute the same function calls but on
 * different data (i.e. different tasks)
 */
template <const char *kernelname, typename Executor> class Aggregated_Executor {
private:
  //===============================================================================
  // Misc private avariables:
  //
  bool slices_exhausted = false;
  const size_t max_slices = 4;
  size_t current_slices = 0;
  std::mutex mut;

  //===============================================================================
  // Subclasses

  /// Slice class - meant as a scope interface to the aggregated executor
  class Executor_Slice {
  private:
    /// Executor is a slice of this aggregated_executor
    Aggregated_Executor<kernelname, Executor> &parent;
    /// Which slice are we?
    const size_t slice_id;
    /// How many slices are there overall - required to check the launch
    /// criteria
    const size_t number_slices;
    /// How many functions have been called - required to enforce sequential
    /// behaviour of kernel launches
    size_t launch_counter{0};

  public:
    Executor_Slice(const size_t number_slices) : number_slices(number_slices) {}
    template <typename F, typename... Ts>
    void post(hpx::lcos::future<void> &stream_future, F &&f, Ts &&...ts) {

      // we should only execute function calls once all slices
      // have been given away (-> Executor Slices start)
      assert(parent.slices_exhausted == true);

      if (parent.function_calls.size() <= launch_counter) {
        // We're the first slice to hit this function call -> create new
        // instance
        parent.function_calls.emplace_back(number_slices);
      }

      // pass function call on the the appropriate object -- increases its
      // counter which will trigger the actual function call once all slices hit
      // it
      parent.function_calls[launch_counter].post(std::forward<F>(f),
                                                 std::forward<Ts>(ts)...);
      // move stream forward
      launch_counter++;
    }
    // TODO async call
  };
  //===============================================================================

  /// Promises with the slice executors -- to be set when the starting criteria
  /// is met
  std::vector<hpx::lcos::local::promise<Executor_Slice>> executor_slices;
  /// List of aggregated function calls - function will be launched when all
  /// slices have called it
  std::vector<aggregated_function_call<kernelname, Executor>> function_calls;

  //===============================================================================
  // Public InterfaceL
public:
  hpx::lcos::future<Executor_Slice> &&request_executor_slice() {
    std::lock_guard<std::mutex> guard(mut);
    if (!slices_exhausted) {
      executor_slices.emplace_back(hpx::lcos::local::promise<Executor_Slice>{});
      current_slices++;
      if (current_slices == 1) {
        // TODO get future and add continuation for when the stream does its
        // thing
      }
      if (current_slices == max_slices) {
        slices_exhausted = true;
        for (auto &slice_promise : executor_slices) {
          // TODO Set slice max number and slice IDs
          slice_promise.set_value(Aggregated_Executor<kernelname, Executor>{});
        }
      }
      return executor_slices.back().get_future();
    } else {
      // TODO call different executor...
      throw "not implemented yet";
    }
  }
};
//===============================================================================
//===============================================================================

// TODO Add buffer aggregation class
// Needs
// - Mutex/flag to stop giving out new slices?
// - std::vector with the futures for slices we gave out
// - Mutex synchronizing access to the vector
// - The Actual buffer...
// - A slice counter
// - Memcpy operations (API) to and from the device
//

int hpx_main(int argc, char *argv[]) {
  // Init method
  {
    std::string filename{};
    try {
      boost::program_options::options_description desc{"Options"};
      desc.add_options()("help", "Help screen");

      boost::program_options::variables_map vm;
      boost::program_options::parsed_options options =
          parse_command_line(argc, argv, desc);
      boost::program_options::store(options, vm);
      boost::program_options::notify(vm);

      if (vm.count("help") == 0u) {
        std::cout << "Running with parameters:" << std::endl << std::endl;
      } else {
        std::cout << desc << std::endl;
        return hpx::finalize();
      }
    } catch (const boost::program_options::error &ex) {
      std::cerr << "CLI argument problem found: " << ex.what() << '\n';
    }
    if (!filename.empty()) {
      freopen(filename.c_str(), "w", stdout); // NOLINT
      freopen(filename.c_str(), "w", stderr); // NOLINT
    }
  }

  stream_pool::init<hpx::cuda::experimental::cuda_executor,
                    round_robin_pool<hpx::cuda::experimental::cuda_executor>>(
      8, 0, false);
  auto executor1 = stream_pool::get_interface<
      hpx::cuda::experimental::cuda_executor,
      round_robin_pool<hpx::cuda::experimental::cuda_executor>>();

  std::vector<hpx::lcos::shared_future<void>> futs;
  hpx::lcos::local::promise<void> promise1{};
  hpx::lcos::local::promise<void> promise2{};
  futs.push_back(promise1.get_shared_future());
  futs.push_back(promise2.get_shared_future());
  futs.push_back(promise1.get_shared_future());
  promise1.set_value();
  std::cout << "beginning when all" << std::endl;
  auto fut = hpx::when_all(futs);
  promise2.set_value();
  auto final_fut = fut.then([&](auto &&old_fut) {
    std::cout << "Such continuation" << std::endl;
    auto futs2 = old_fut.get();
    for (int i = 0; i < futs.size(); i++) {
      futs2[i].get();
      std::cout << "Got future " << i + 1
                << std::endl; // futs2[i].get() << std::endl;
    }
  });
  hpx::lcos::local::latch latch1{4};
  while (!latch1.is_ready()) {
    std::cout << "Counting down..." << std::endl;
    latch1.count_down(1);
  }
  std::cout << "before get" << std::endl;
  final_fut.get();
  latch1.count_up(1);
  // waiting only stops when we hit 0 (going negativ is bad..)
  latch1.arrive_and_wait();

  static const char hello17[] = "C++17 [a-zA-Z]*";
  aggregated_function_call<hello17, decltype(executor1)> bla{4};
  hpx::lcos::local::promise<void> trigger{};
  auto trigger_fut = trigger.get_future();
  bla.post_when(trigger_fut, print_stuff3, 33);
  bla.post_when(trigger_fut, print_stuff3, 34);
  bla.post_when(trigger_fut, print_stuff3, 35);

  function_call_aggregator test{};
  // test.add_continuation();
  test.post(print_stuff1, 1);
  test.post(print_stuff1, 2);
  test.post(print_stuff1, 3);
  test.post(print_stuff1, 4);
  test.post(print_stuff2, 5, 3.1);

  std::cin.get();
  std::cout << " Fulfilling launch promise..." << std::endl;
  test.launch_promise.set_value();
  trigger.set_value();
  std::cin.get();
  std::cout << " Fulfilling slices promise..." << std::endl;
  test.slices_ready_promise.set_value();
  std::cin.get();
  std::cout << " Requesting function future explicitly..." << std::endl;
  bla.post_when(trigger_fut, print_stuff3, 36);
  trigger_fut.get();
  std::cin.get();
  std::cout << " Requesting final future explicitly..." << std::endl;
  test.current_future.get();
  std::cin.get();
  // comparison
  recycler::force_cleanup(); // Cleanup all buffers and the managers for better
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
