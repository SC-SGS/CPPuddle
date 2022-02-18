// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <any>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <hpx/futures/future.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <typeinfo>

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/promise.hpp>

#include <hpx/async_cuda/cuda_executor.hpp>

#include <boost/core/demangle.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"
#include "../include/stream_manager.hpp"

#include <stdio.h>
//===============================================================================
//===============================================================================
// Helper classes

/// Helper class for the helper class that prints tuples -- do not use this
/// directly
template <class TupType, size_t... I>
void print_tuple(const TupType &_tup, std::index_sequence<I...>) {
  (..., (hpx::cout << (I == 0 ? "" : ", ") << std::get<I + 1>(_tup)));
}

/// Helper class for printing tuples (first component should be a function
/// pointer, remaining components the function arguments)
template <class... T> void print_tuple(const std::tuple<T...> &_tup) {
  // Use pointer and sprintf as boost::format refused to NOT cast the pointer
  // address to 1...
  // TODO Try using std::format as soon as we can move to C++20
  // TODO Put on stack?
  std::unique_ptr<char[]> debug_string(new char[128]());
  snprintf(debug_string.get(), 128, "Function address: %p -- Arguments: (",
           std::get<0>(_tup));
  hpx::cout << debug_string.get();
  print_tuple(_tup, std::make_index_sequence<sizeof...(T) - 1>());
  hpx::cout << ")";
}

//===============================================================================
//===============================================================================
// Example functions

void print_stuff_error(int used_slices, int i) {
  hpx::cout << "i is not " << i << "(Slice " << used_slices << ")" << std::endl;
}
void print_stuff1(int used_slices, int i) {
  hpx::cout << "i is " << i << "(Slice " << used_slices << ")" << std::endl;
}
void print_stuff2(int used_slices, int i, double d) {
  hpx::cout << "i is " << i << std::endl;
  hpx::cout << "d is " << d << std::endl;
  hpx::cout << "(Slice is " << used_slices << ")" << std::endl;
}
void print_stuff3(const std::atomic<int> &used_slices, int i) {
  const int bla = used_slices;
  hpx::cout << "i is " << i << "(Slice " << bla << ")" << std::endl;
}
/*void print_stuff1(int *used_slices, int i) {
  hpx::cout << "i is " << i << "(Slice " << *used_slices << ")" << std::endl;
}
void print_stuff2(int *used_slices, int i, double d) {
  hpx::cout << "i is " << i << std::endl;
  hpx::cout << "d is " << d << std::endl;
  hpx::cout << "(Slice is " << *used_slices << ")" << std::endl;
}*/

//===============================================================================
//===============================================================================
/// obsolete class - kept for reference
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
                           hpx::cout << "starting function continuation"
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
    if (f==f && args == args) hpx::cout << "yay" << std::endl;
    current_future = hpx::lcos::when_all(current_future, all_slices_ready).then(
        [f = std::move(f), args = std::move(args)](auto &&old_fut) {
          hpx::cout << "starting function continuation" << std::endl;
          std::apply(f, std::move(args));
          return;
        });
    this->current_slice_number++;
  }*/
};

//===============================================================================
//===============================================================================
/// Manages the launch conditions for aggregated function calls
/// type/value-errors
/** Launch conditions: All slice executors must have called the same function
 * (tracked by future all_slices_ready)
 * AND
 * Previous aggregated_function_call on the same Executor must have been
 * launched (tracked by future stream_future)
 * All function calls received from the slice executors are checked if they
 * match the first one in both types and values (throws exception otherwise)
 */

template <const char *kernelname, typename Executor>
class aggregated_function_call {
private:
  std::atomic<int> slice_counter = 0; // TODO not necessasry?

  /// Promise to be set when all slices have visited this function call
  hpx::lcos::local::promise<void> slices_ready_promise;
  /// Tracks if all slices have visited this function call
  hpx::lcos::future<void> all_slices_ready = slices_ready_promise.get_future();
  /// How many slices can we expect?
  const size_t number_slices;

  /// Stores the function call of the first slice as reference for error checking
  std::any function_tuple;
  /// Stores the string of the first function call for debug output
  std::string debug_type_information;

public:
  aggregated_function_call(const size_t number_slices)
      : number_slices(number_slices) {}
  template <typename F, typename... Ts>
  void post_when(hpx::lcos::future<void> &stream_future, F &&f, Ts &&...ts) {
    slice_counter++;
    std::atomic<int> &current_slice_counter = this->slice_counter;
    hpx::cout << "Slices counter ... " << slice_counter << std::endl;

    if (slice_counter == 1) {
      stream_future = hpx::lcos::when_all(stream_future, all_slices_ready)
                          .then([=, &current_slice_counter](auto &&old_fut) {
                            hpx::cout << "starting function_call continuation"
                                      << std::endl;
                            // TODO modify according to slices (launch either X
                            // seperate ones, or have one specialization
                            // launching stuff...)
                            f(current_slice_counter, ts...);
                            hpx::cout << kernelname << std::endl;
                            ;
                            return;
                          });
      auto tmp_tuple = std::make_tuple(f, std::forward<Ts>(ts)...);
      function_tuple = tmp_tuple;
      debug_type_information = typeid(decltype(tmp_tuple)).name();

    } else {
      //
      // This scope checks if both the type and the values of the current call
      // match the original call To be used in debug build...
      //
      // TODO Only enable error checking without NDEBUG (avoiding performance
      // penalities in Release build
      auto comparison_tuple = std::make_tuple(f, std::forward<Ts>(ts)...);
      try {
        auto orig_call_tuple =
            std::any_cast<decltype(comparison_tuple)>(function_tuple);
        if (comparison_tuple != orig_call_tuple) {
          throw std::runtime_error("Values of function arguments (or function "
                                   "itself) do not match ");
        }
      } catch (const std::bad_any_cast &e) {
        hpx::cout << "\nMismatched types error in aggregated call of executor "
                  << kernelname << ": " << e.what() << "\n";
        hpx::cout << "Expected types:\t\t "
                  << boost::core::demangle(debug_type_information.c_str())
                  << "\n";
        hpx::cout << "Got types:\t\t "
                  << boost::core::demangle(
                         typeid(decltype(comparison_tuple)).name())
                  << "\n";
        hpx::cout << std::endl;
        throw;
      } catch (const std::runtime_error &e) {
        hpx::cout << "\nMismatched values error in aggregated call of executor "
                  << kernelname << ": " << e.what() << std::endl;
        hpx::cout << "Types (matched):\t "
                  << boost::core::demangle(debug_type_information.c_str())
                  << "\n";
        auto orig_call_tuple =
            std::any_cast<decltype(comparison_tuple)>(function_tuple);
        hpx::cout << "Expected values:\t ";
        print_tuple(orig_call_tuple);
        hpx::cout << "\n";
        hpx::cout << "Got values:\t\t ";
        print_tuple(comparison_tuple);
        hpx::cout << std::endl << std::endl;
        ;
        throw;
      }
    }
    // Check exit criteria: Launch function call continuation by setting the
    // slices promise
    if (slice_counter == number_slices) {
      hpx::cout << "Starting slices ready..." << std::endl;
      slices_ready_promise.set_value();
    }
  }
  // TODO async call
  // We need to be able to copy or no-except move for std::vector..
  aggregated_function_call(const aggregated_function_call &other) = default;
  aggregated_function_call &
  operator=(const aggregated_function_call &other) = default;
  aggregated_function_call(aggregated_function_call &&other) = default;
  aggregated_function_call &
  operator=(aggregated_function_call &&other) = default;
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
  bool slices_exhausted;
  const size_t max_slices;
  size_t current_slices;
  std::mutex mut;

  //===============================================================================
  // Subclasses

  /// Slice class - meant as a scope interface to the aggregated executor
  class Executor_Slice {
  private:
    /// Executor is a slice of this aggregated_executor
    Aggregated_Executor<kernelname, Executor> &parent;
    /// How many slices are there overall - required to check the launch
    /// criteria
    const size_t number_slices;
    /// How many functions have been called - required to enforce sequential
    /// behaviour of kernel launches
    size_t launch_counter{0};

  public:
    Executor_Slice(Aggregated_Executor &parent, const size_t number_slices)
        : parent(parent), number_slices(number_slices) {}
    template <typename F, typename... Ts> void post(F &&f, Ts &&...ts) {

      // we should only execute function calls once all slices
      // have been given away (-> Executor Slices start)
      assert(parent.slices_exhausted == true);

      parent.post(launch_counter, std::forward<F>(f), std::forward<Ts>(ts)...);
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
  std::list<aggregated_function_call<kernelname, Executor>> function_calls;

  //===============================================================================
  // Public Interface
public:
  hpx::lcos::local::promise<void> dummy_stream_promise;
  hpx::lcos::future<void> current_continuation;
  hpx::lcos::future<void> last_stream_launch_done;

  /// Only meant to be accessed by the slice executors
  template <typename F, typename... Ts>
  void post(const size_t slice_launch_counter, F &&f, Ts &&...ts) {
    // std::lock_guard<std::mutex> guard(mut);

    // Add function call object in case it hasn't happened for this launch yet
    if (function_calls.size() <= slice_launch_counter) {
      std::lock_guard<std::mutex> guard(mut);
      if (function_calls.size() <= slice_launch_counter) {
        function_calls.emplace_back(current_slices);
      }
    }

    // as we cannot copy or non-except move the function objects: use list
    auto it = function_calls.begin();
    std::advance(it, slice_launch_counter);
    it->post_when(last_stream_launch_done, std::forward<F>(f),
                  std::forward<Ts>(ts)...);
  }

  hpx::lcos::future<Executor_Slice> request_executor_slice() {
    hpx::cout << "Trying to lock requestor..." << std::endl;
    std::lock_guard<std::mutex> guard(mut);
    hpx::cout << "Requestor locked" << std::endl;
    if (!slices_exhausted) {
      executor_slices.emplace_back(hpx::lcos::local::promise<Executor_Slice>{});
      hpx::lcos::future<Executor_Slice> ret_fut =
          executor_slices.back().get_future();

      current_slices++;
      if (current_slices == 1) {
        // TODO get future and add continuation for when the stream does its
        // thing
        auto fut = dummy_stream_promise.get_future();
        current_continuation = fut.then([this,
                                         kernelname = kernelname](auto &&fut) {
          hpx::cout << "Trying to lock continuation in " << kernelname << " ..."
                    << std::endl;
          std::lock_guard<std::mutex> guard(mut);
          hpx::cout << "Continuation locked" << std::endl;
          if (!slices_exhausted) {
            slices_exhausted = true;
            for (auto &slice_promise : executor_slices) {
              slice_promise.set_value(Executor_Slice{*this, current_slices});
            }
            executor_slices.clear();
          }
          hpx::cout << "Releasing continuation" << std::endl;
        });
      }
      if (current_slices == max_slices) {
        slices_exhausted = true;
        for (auto &slice_promise : executor_slices) {
          slice_promise.set_value(Executor_Slice{*this, current_slices});
        }
        executor_slices.clear();
      }
      hpx::cout << "Releasing requestor" << std::endl;
      return ret_fut;
    } else {
      // TODO call different executor...
      hpx::cout << "This should not happen!" << std::endl;
      throw "not implemented yet";
    }
  }
  ~Aggregated_Executor(void) {
    // Finish the continuation to not leave a dangling task!
    // otherwise the continuation might access data of non-existent object...
    current_continuation.get();
  }

  Aggregated_Executor(const size_t number_slices)
      : max_slices(number_slices), current_slices(0), slices_exhausted(false),
        current_continuation(hpx::lcos::make_ready_future()),
        last_stream_launch_done(hpx::lcos::make_ready_future()) {}
  // Not meant to be copied or moved
  Aggregated_Executor(const Aggregated_Executor &other) = delete;
  Aggregated_Executor &operator=(const Aggregated_Executor &other) = delete;
  Aggregated_Executor(Aggregated_Executor &&other) = delete;
  Aggregated_Executor &operator=(Aggregated_Executor &&other) = delete;
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
        hpx::cout << "Running with parameters:" << std::endl << std::endl;
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

  stream_pool::init<hpx::cuda::experimental::cuda_executor,
                    round_robin_pool<hpx::cuda::experimental::cuda_executor>>(
      8, 0, false);
  auto executor1 = stream_pool::get_interface<
      hpx::cuda::experimental::cuda_executor,
      round_robin_pool<hpx::cuda::experimental::cuda_executor>>();

  // Sequential test
  hpx::cout << "Sequential test with all executor slices" << std::endl;
  hpx::cout << "----------------------------------------" << std::endl;
  {
    static const char kernelname1[] = "Dummy Kernel 1";
    Aggregated_Executor<kernelname1, decltype(executor1)> agg_exec{4};

    auto slice_fut1 = agg_exec.request_executor_slice();

    std::vector<hpx::lcos::future<void>> slices_done_futs;
    slices_done_futs.emplace_back(slice_fut1.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 1" << std::endl;
      slice_exec.post(print_stuff1, 1);
      slice_exec.post(print_stuff2, 1, 1.0);
    }));

    auto slice_fut2 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut2.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 2" << std::endl;
      slice_exec.post(print_stuff1, 1);
      slice_exec.post(print_stuff2, 1, 1.0);
    }));

    auto slice_fut3 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut3.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 3" << std::endl;
      slice_exec.post(print_stuff1, 1);
      slice_exec.post(print_stuff2, 1, 1.0);
    }));

    auto slice_fut4 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut4.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 4" << std::endl;
      slice_exec.post(print_stuff1, 1);
      slice_exec.post(print_stuff2, 1, 1.0);
    }));
    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by equesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
    if (slices_done_futs[3].has_exception())
      throw "bla";
    if (final_fut.has_exception())
      throw "shit";

    agg_exec.dummy_stream_promise.set_value();
  }
  hpx::cout << std::endl;
  // std::cin.get();

  // Interruption test
  hpx::cout << "Sequential test with interruption:" << std::endl;
  hpx::cout << "----------------------------------" << std::endl;
  {
    static const char kernelname1[] = "Dummy Kernel 2";
    Aggregated_Executor<kernelname1, decltype(executor1)> agg_exec{4};

    auto slice_fut1 = agg_exec.request_executor_slice();

    std::vector<hpx::lcos::future<void>> slices_done_futs;
    slices_done_futs.emplace_back(slice_fut1.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 1" << std::endl;
    }));

    auto slice_fut2 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut2.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 2" << std::endl;
    }));

    auto slice_fut3 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut3.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 3" << std::endl;
    }));

    hpx::cout << "Requested 3 executors!" << std::endl;
    hpx::cout << "Realizing by setting the continuation future..." << std::endl;
    agg_exec.dummy_stream_promise.set_value();
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
  }
  hpx::cout << std::endl;

  // Error test
  hpx::cout << "Error test with all wrong types and values in 2 slices"
            << std::endl;
  hpx::cout << "------------------------------------------------------"
            << std::endl;
  {
    static const char kernelname1[] = "Dummy Kernel 1";
    Aggregated_Executor<kernelname1, decltype(executor1)> agg_exec{4};

    auto slice_fut1 = agg_exec.request_executor_slice();

    std::vector<hpx::lcos::future<void>> slices_done_futs;
    slices_done_futs.emplace_back(slice_fut1.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 1" << std::endl;
      slice_exec.post(print_stuff1, 3);
    }));

    auto slice_fut2 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut2.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 2" << std::endl;
      slice_exec.post(print_stuff1, 3);
    }));

    auto slice_fut3 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut3.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 3" << std::endl;
      try {
        slice_exec.post(print_stuff1, 3.0f);
      } catch (...) {
        hpx::cout << "TEST succeeded: Found type error exception!\n"
                  << std::endl;
      }
    }));

    auto slice_fut4 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut4.then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 4" << std::endl;
      // TODO How to propagate the exception?
      try {
        slice_exec.post(print_stuff_error, 3);
      } catch (...) {
        hpx::cout << "TEST succeeded: Found value error exception!\n"
                  << std::endl;
      }
    }));

    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by equesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
    agg_exec.dummy_stream_promise.set_value();
  }
  hpx::cout << std::endl;
  hpx::cout << "Done!" << std::endl;

  recycler::force_cleanup(); // Cleanup all buffers and the managers for better
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
