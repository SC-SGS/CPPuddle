// Copyright (c) 2022-2022 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <stdio.h>

#include <any>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <hpx/synchronization/mutex.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

#undef NDEBUG
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
#include <hpx/futures/future.hpp>

//===============================================================================
//===============================================================================
// Helper functions/classes

/// Constructs a tuple with copies (to store temporaries in aggregated function
/// calls) yet also supporting references (on the users own risk...)
template <typename... Ts>
std::tuple<Ts...> make_tuple_supporting_references(Ts &&...ts) {
  return std::tuple<Ts...>{std::forward<Ts>(ts)...};
}

/// Print some specific values that we can, but don't bother for most types
/// (such as vector)
template <typename T> std::string print_if_possible(T val) {
  if constexpr (std::is_convertible_v<T, std::string>) {
    return val;
  } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
    return std::to_string(val);
  } else if constexpr (std::is_pointer_v<T>) {
    // Pretty printing pointer sort of only works well with %p
    // TODO Try using std::format as soon as we can move to C++20
    std::unique_ptr<char[]> debug_string(new char[128]());
    snprintf(debug_string.get(), 128, "%p", val);
    return std::string(debug_string.get());
  } else {
    return std::string("cannot print value");
  }
}

/// Helper class for the helper class that prints tuples -- do not use this
/// directly
template <class TupType, size_t... I>
void print_tuple(const TupType &_tup, std::index_sequence<I...>) {
  (..., (hpx::cout << (I == 0 ? "" : ", ")
                   << print_if_possible(std::get<I + 1>(_tup))));
}

/// Helper class for printing tuples (first component should be a function
/// pointer, remaining components the function arguments)
template <class... T> void print_tuple(const std::tuple<T...> &_tup) {
  // Use pointer and sprintf as boost::format refused to NOT cast the pointer
  // address to 1...
  // TODO Try using std::format as soon as we can move to C++20
  std::unique_ptr<char[]> debug_string(new char[128]());
  snprintf(debug_string.get(), 128, "Function address: %p -- Arguments: (",
           std::get<0>(_tup));
  hpx::cout << debug_string.get();
  print_tuple(_tup, std::make_index_sequence<sizeof...(T) - 1>());
  hpx::cout << ")";
}

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
// Example functions

void print_stuff_error(int i) { hpx::cout << "i is not " << i << std::endl; }
void print_stuff1(int i) { hpx::cout << "i is " << i << std::endl; }
void print_stuff2(int i, double d) {
  hpx::cout << "i is " << i << std::endl;
  hpx::cout << "d is " << d << std::endl;
}
void print_stuff3(int i) { hpx::cout << "i is " << i << std::endl; }

size_t add_pointer_launches = 0.0;
template <typename T>
void add_pointer(size_t aggregation_size, T *A, T *B, T *C) {
  add_pointer_launches++;
  const size_t start_id = 0;
  for (size_t i = 0; i < aggregation_size; i++) {
    C[start_id + i] = B[start_id + i] + A[start_id + i];
  }
}

size_t add_launches = 0.0;
template <typename Container>
void add(size_t slice_size, Container &A, Container &B, Container &C) {
  add_launches++;
  const size_t start_id = 0;
  for (size_t i = 0; i < 4 * slice_size; i++) {
    C[start_id + i] = B[start_id + i] + A[start_id + i];
  }
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

template <typename Executor> class aggregated_function_call {
private:
  std::atomic<size_t> slice_counter = 0;

  /// Promise to be set when all slices have visited this function call
  hpx::lcos::local::promise<void> slices_ready_promise;
  /// Tracks if all slices have visited this function call
  hpx::lcos::future<void> all_slices_ready = slices_ready_promise.get_future();
  /// How many slices can we expect?
  const size_t number_slices;
  const bool async_mode;

#ifndef NDEBUG
#pragma message                                                                \
    "Running slow work aggegator debug build! Run with NDEBUG defined for fast build..."
  /// Stores the function call of the first slice as reference for error
  /// checking
  std::any function_tuple;
  /// Stores the string of the first function call for debug output
  std::string debug_type_information;
  std::mutex debug_mut;
#endif

  std::vector<hpx::lcos::local::promise<void>> potential_async_promises{};

public:
  aggregated_function_call(const size_t number_slices, bool async_mode)
      : number_slices(number_slices), async_mode(async_mode) {
    if (async_mode)
      potential_async_promises.resize(number_slices);
  }
  ~aggregated_function_call(void) {
    // All slices should have done this call
    assert(slice_counter == number_slices);
    assert(!all_slices_ready.valid());
  }
  template <typename F, typename... Ts>
  void post_when(hpx::lcos::future<void> &stream_future, F &&f, Ts &&...ts) {
#ifndef NDEBUG
    // needed for concurrent access to function_tuple and debug_type_information
    // Not required for normal use
    std::lock_guard<std::mutex> guard(debug_mut);
#endif
    assert(!async_mode);
    assert(potential_async_promises.empty());
    const size_t local_counter = slice_counter++;

    // auto args_tuple(
    if (local_counter == 0) {
      std::tuple<Ts...> args =
          make_tuple_supporting_references(std::forward<Ts>(ts)...);
      stream_future =
          hpx::lcos::when_all(stream_future, all_slices_ready)
              .then([f, args = std::move(args)](auto &&old_fut) mutable {
                std::apply(f, std::move(args));
                return;
              });
#ifndef NDEBUG
      auto tmp_tuple =
          make_tuple_supporting_references(f, std::forward<Ts>(ts)...);
      function_tuple = tmp_tuple;
      debug_type_information = typeid(decltype(tmp_tuple)).name();
#endif

    } else {
      //
      // This scope checks if both the type and the values of the current call
      // match the original call To be used in debug build...
      //
#ifndef NDEBUG
      auto comparison_tuple =
          make_tuple_supporting_references(f, std::forward<Ts>(ts)...);
      try {
        auto orig_call_tuple =
            std::any_cast<decltype(comparison_tuple)>(function_tuple);
        if (comparison_tuple != orig_call_tuple) {
          throw std::runtime_error(
              "Values of post function arguments (or function "
              "itself) do not match ");
        }
      } catch (const std::bad_any_cast &e) {
        std::cerr
            << "\nMismatched types error in aggregated post call of executor "
            << ": " << e.what() << "\n";
        std::cerr << "Expected types:\t\t "
                  << boost::core::demangle(debug_type_information.c_str());
        std::cerr << "\nGot types:\t\t "
                  << boost::core::demangle(
                         typeid(decltype(comparison_tuple)).name())
                  << "\n"
                  << std::endl;
        throw;
      } catch (const std::runtime_error &e) {
        std::cerr
            << "\nMismatched values error in aggregated post call of executor "
            << ": " << e.what() << std::endl;
        std::cerr << "Types (matched):\t "
                  << boost::core::demangle(debug_type_information.c_str());
        auto orig_call_tuple =
            std::any_cast<decltype(comparison_tuple)>(function_tuple);
        std::cerr << "\nExpected values:\t ";
        print_tuple(orig_call_tuple);
        std::cerr << "\nGot values:\t\t ";
        print_tuple(comparison_tuple);
        std::cerr << std::endl << std::endl;
        throw;
      }
#endif
    }
    assert(local_counter < number_slices);
    assert(slice_counter < number_slices + 1);
    // Check exit criteria: Launch function call continuation by setting the
    // slices promise
    if (local_counter == number_slices - 1) {
      slices_ready_promise.set_value();
    }
  }
  template <typename F, typename... Ts>
  hpx::lcos::future<void> async_when(hpx::lcos::future<void> &stream_future,
                                     F &&f, Ts &&...ts) {
#ifndef NDEBUG
    // needed for concurrent access to function_tuple and debug_type_information
    // Not required for normal use
    std::lock_guard<std::mutex> guard(debug_mut);
#endif
    assert(async_mode);
    assert(!potential_async_promises.empty());
    const size_t local_counter = slice_counter++;
    if (local_counter == 0) {
      std::tuple<Ts...> args =
          make_tuple_supporting_references(std::forward<Ts>(ts)...);
      std::vector<hpx::lcos::local::promise<void>> &potential_async_promises =
          this->potential_async_promises;
      stream_future =
          hpx::lcos::when_all(stream_future, all_slices_ready)
              .then([f, args = std::move(args),
                     &potential_async_promises](auto &&old_fut) mutable {
                std::apply(f, std::move(args));
                hpx::lcos::future<void> fut = hpx::lcos::make_ready_future();
                fut.then([&potential_async_promises](auto &&fut) {
                  for (auto &promise : potential_async_promises) {
                    promise.set_value();
                  }
                });
                return;
              });
#ifndef NDEBUG
      auto tmp_tuple =
          make_tuple_supporting_references(f, std::forward<Ts>(ts)...);
      function_tuple = tmp_tuple;
      debug_type_information = typeid(decltype(tmp_tuple)).name();
#endif

    } else {
      //
      // This scope checks if both the type and the values of the current call
      // match the original call To be used in debug build...
      //
#ifndef NDEBUG
      auto comparison_tuple =
          make_tuple_supporting_references(f, std::forward<Ts>(ts)...);
      try {
        auto orig_call_tuple =
            std::any_cast<decltype(comparison_tuple)>(function_tuple);
        if (comparison_tuple != orig_call_tuple) {
          throw std::runtime_error(
              "Values of async function arguments (or function "
              "itself) do not match ");
        }
      } catch (const std::bad_any_cast &e) {
        std::cerr
            << "\nMismatched types error in aggregated async call of executor "
            << ": " << e.what() << "\n";
        std::cerr << "Expected types:\t\t "
                  << boost::core::demangle(debug_type_information.c_str());
        std::cerr << "\nGot types:\t\t "
                  << boost::core::demangle(
                         typeid(decltype(comparison_tuple)).name())
                  << "\n"
                  << std::endl;
        throw;
      } catch (const std::runtime_error &e) {
        std::cerr
            << "\nMismatched values error in aggregated async call of executor "
            << ": " << e.what() << std::endl;
        std::cerr << "Types (matched):\t "
                  << boost::core::demangle(debug_type_information.c_str());
        auto orig_call_tuple =
            std::any_cast<decltype(comparison_tuple)>(function_tuple);
        std::cerr << "\nExpected values:\t ";
        print_tuple(orig_call_tuple);
        std::cerr << "\nGot values:\t\t ";
        print_tuple(comparison_tuple);
        std::cerr << std::endl << std::endl;
        throw;
      }
#endif
    }
    assert(local_counter < number_slices);
    assert(slice_counter < number_slices + 1);
    assert(potential_async_promises.size() == number_slices);
    hpx::lcos::future<void> ret_fut =
        potential_async_promises[local_counter].get_future();
    // Check exit criteria: Launch function call continuation by setting the
    // slices promise
    if (local_counter == number_slices - 1) {
      slices_ready_promise.set_value();
    }
    return ret_fut;
  }
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

enum class Aggregated_Executor_Modes { EAGER = 1, STRICT, ENDLESS };
/// Declaration since the actual allocator is only defined after the Executors
template <typename T, typename Host_Allocator, typename Executor>
class Allocator_Slice;

/// Executor Class that aggregates function calls for specific kernels
/** Executor is not meant to be used directly. Instead it yields multiple
 * Executor_Slice objects. These serve as interfaces. Slices from the same
 * Aggregated_Executor are meant to execute the same function calls but on
 * different data (i.e. different tasks)
 */
template <typename Executor> class Aggregated_Executor {
private:
  //===============================================================================
  // Misc private avariables:
  //
  bool slices_exhausted;
  const Aggregated_Executor_Modes mode;
  const size_t max_slices;
  size_t current_slices;
  Executor executor;

public:
  // Subclasses

  /// Slice class - meant as a scope interface to the aggregated executor
  class Executor_Slice {
  private:
    /// Executor is a slice of this aggregated_executor
    Aggregated_Executor<Executor> &parent;
    /// How many functions have been called - required to enforce sequential
    /// behaviour of kernel launches
    size_t launch_counter{0};
    size_t buffer_counter{0};
    bool notify_parent_about_destruction{true};

  public:
    /// How many slices are there overall - required to check the launch
    /// criteria
    const size_t number_slices;
    const size_t id;
    using executor_t = Executor;
    Executor_Slice(Aggregated_Executor &parent, const size_t slice_id,
                   const size_t number_slices)
        : parent(parent), notify_parent_about_destruction(true),
          number_slices(number_slices), id(slice_id) {}
    ~Executor_Slice(void) {

      // Don't notify parent if we moved away from this executor_slice
      if (notify_parent_about_destruction) {
        // Executor should be done by the time of destruction
        assert(buffer_counter == 0);
        assert(launch_counter == parent.function_calls.size());
        // Notifiy parent that this aggregation slice is one
        parent.reduce_usage_counter();
      }
    }
    Executor_Slice(const Executor_Slice &other) = delete;
    Executor_Slice &operator=(const Executor_Slice &other) = delete;
    Executor_Slice(Executor_Slice &&other)
        : parent(other.parent), launch_counter(std::move(other.launch_counter)),
          buffer_counter(std::move(other.buffer_counter)),
          number_slices(std::move(other.number_slices)),
          id(std::move(other.id)) {
      other.notify_parent_about_destruction = false;
    }
    Executor_Slice &operator=(Executor_Slice &&other) {
      parent = other.parent;
      launch_counter = std::move(other.launch_counter);
      buffer_counter = std::move(other.buffer_counter);
      number_slices = std::move(other.number_slices);
      id = std::move(other.id);
      other.notify_parent_about_destruction = false;
    }
    template <typename T, typename Host_Allocator>
    Allocator_Slice<T, Host_Allocator, Executor> make_allocator() {
      return Allocator_Slice<T, Host_Allocator, Executor>(*this);
    }
    template <typename F, typename... Ts> void post(F &&f, Ts &&...ts) {
      // we should only execute function calls once all slices
      // have been given away (-> Executor Slices start)
      assert(parent.slices_exhausted == true);
      parent.post(launch_counter, std::forward<F>(f), std::forward<Ts>(ts)...);
      launch_counter++;
    }
    template <typename F, typename... Ts>
    hpx::lcos::future<void> async(F &&f, Ts &&...ts) {
      // we should only execute function calls once all slices
      // have been given away (-> Executor Slices start)
      assert(parent.slices_exhausted == true);
      hpx::lcos::future<void> ret_fut = parent.async(
          launch_counter, std::forward<F>(f), std::forward<Ts>(ts)...);
      launch_counter++;
      return ret_fut;
    }

    /// Get new aggregated buffer (might have already been allocated been
    /// allocated by different slice)
    template <typename T, typename Host_Allocator> T *get(const size_t size) {
      assert(parent.slices_exhausted == true);
      T *aggregated_buffer =
          parent.get<T, Host_Allocator>(size, buffer_counter);
      buffer_counter++;
      return aggregated_buffer;
    }
    /// Mark aggregated buffer as unused (should only happen due to stack
    /// unwinding) Do not call manually
    template <typename T, typename Host_Allocator>
    void mark_unused(T *p, const size_t size) {
      assert(parent.slices_exhausted == true);
      buffer_counter--;
      parent.mark_unused<T, Host_Allocator>(p, size, buffer_counter);
    }
    // TODO Support the reference counting used previousely in CPPuddle?
    // might be required for Kokkos
  };

  //===============================================================================

  /// Promises with the slice executors -- to be set when the starting criteria
  /// is met
  std::vector<hpx::lcos::local::promise<Executor_Slice>> executor_slices;
  /// List of aggregated function calls - function will be launched when all
  /// slices have called it
  std::deque<aggregated_function_call<Executor>> function_calls;
  /// For synchronizing the access to the function calls list
  std::mutex mut;

  // TODO Evaluate if we should switch to boost::any and unsafe casts (=> should
  // be faster)
  /// Data entry for a buffer allocation: any for the point, size_t for
  /// buffer-size, atomic for the slice counter
  using buffer_entry_t =
      std::tuple<std::any, const size_t, std::atomic<size_t>, bool>;
  /// Keeps track of the aggregated buffer allocations done in all the slices
  std::deque<buffer_entry_t> buffer_allocations;
  /// For synchronizing the access to the buffer_allocations
  std::mutex buffer_mut;

  /// Get new buffer OR get buffer already allocated by different slice
  template <typename T, typename Host_Allocator>
  T *get(const size_t size, const size_t slice_alloc_counter) {
    assert(slices_exhausted == true);
    // Add aggreated buffer entry in case it hasn't happened yet for this call
    // First: Check if it already has happened
    if (buffer_allocations.size() <= slice_alloc_counter) {
      // we might be the first! Lock...
      std::lock_guard<std::mutex> guard(buffer_mut);
      // ... and recheck
      if (buffer_allocations.size() <= slice_alloc_counter) {
        // Get shiny and new buffer that will be shared between all slices
        // Buffer might be recycled from previous allocations by the
        // buffer_recycler...
        T *aggregated_buffer =
            recycler::detail::buffer_recycler::get<T, Host_Allocator>(size,
                                                                      true);
        // Create buffer entry for this buffer
        buffer_allocations.emplace_back(aggregated_buffer, size, 1, true);

        // Return buffer
        return aggregated_buffer;
      }
    }
    assert(std::get<2>(buffer_allocations[slice_alloc_counter]) >= 1);

    // Buffer entry should already exist:
    T *aggregated_buffer = std::any_cast<T *>(
        std::get<0>(buffer_allocations[slice_alloc_counter]));
    // Error handling: Size is wrong?
    assert(size == std::get<1>(buffer_allocations[slice_alloc_counter]));
    /*if (size != std::get<1>(buffer_allocations[slice_alloc_counter])) {
      throw std::runtime_error("Requested buffer size does not match the size "
                               "in the aggregated buffer record!");
    }*/
    // Notify that one more slice has visited this buffer allocation
    std::get<2>(buffer_allocations[slice_alloc_counter])++;
    return aggregated_buffer;
  }

  /// Notify buffer list that one slice is done with the buffer
  template <typename T, typename Host_Allocator>
  void mark_unused(T *p, const size_t size, const size_t slice_alloc_counter) {
    assert(slices_exhausted == true);
    assert(slice_alloc_counter < buffer_allocations.size());
    auto &[buffer_pointer_any, buffer_size, buffer_allocation_counter, valid] =
        buffer_allocations[slice_alloc_counter];
    T *buffer_pointer = std::any_cast<T *>(buffer_pointer_any);

    assert(buffer_size == size);
    assert(p == buffer_pointer);
    // assert(buffer_pointer == p || buffer_pointer == nullptr);
    // Slice is done with this buffer
    buffer_allocation_counter--;
    // Check if all slices are done with this buffer?
    if (buffer_allocation_counter == 0) {
      // Yes! "Deallocate" by telling the recylcer the buffer is fit for reusage
      std::lock_guard<std::mutex> guard(buffer_mut);
      // Only mark unused if another buffer has not done so already (and marked
      // it as invalid)
      if (valid) {
        recycler::detail::buffer_recycler::mark_unused<T, Host_Allocator>(
            buffer_pointer, buffer_size);
        // mark buffer as invalid to prevent any other slice from marking the
        // buffer as unused
        valid = false;
      }
    }
  }
  // TODO Support the reference counting used previousely in CPPuddle?
  // might be required for Kokkos

  //===============================================================================
  // Public Interface
public:
  hpx::lcos::future<void> current_continuation;
  hpx::lcos::future<void> last_stream_launch_done;

  /// Only meant to be accessed by the slice executors
  template <typename F, typename... Ts>
  void post(const size_t slice_launch_counter, F &&f, Ts &&...ts) {
    assert(slices_exhausted == true);
    // Add function call object in case it hasn't happened for this launch yet
    if (function_calls.size() <= slice_launch_counter) {
      std::lock_guard<std::mutex> guard(mut);
      if (function_calls.size() <= slice_launch_counter) {
        function_calls.emplace_back(current_slices, false);
      }
    }

    function_calls[slice_launch_counter].post_when(
        last_stream_launch_done, std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  /// Only meant to be accessed by the slice executors
  template <typename F, typename... Ts>
  hpx::lcos::future<void> async(const size_t slice_launch_counter, F &&f,
                                Ts &&...ts) {
    assert(slices_exhausted == true);
    // Add function call object in case it hasn't happened for this launch yet
    if (function_calls.size() <= slice_launch_counter) {
      std::lock_guard<std::mutex> guard(mut);
      if (function_calls.size() <= slice_launch_counter) {
        function_calls.emplace_back(current_slices, true);
      }
    }

    return function_calls[slice_launch_counter].async_when(
        last_stream_launch_done, std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  bool slice_available(void) {
    std::lock_guard<std::mutex> guard(mut);
    return !slices_exhausted;
  }

  std::optional<hpx::lcos::future<Executor_Slice>> request_executor_slice() {
    std::lock_guard<std::mutex> guard(mut);
    if (!slices_exhausted) {
      executor_slices.emplace_back(hpx::lcos::local::promise<Executor_Slice>{});
      hpx::lcos::future<Executor_Slice> ret_fut =
          executor_slices.back().get_future();

      current_slices++;
      if (current_slices == 1 && mode == Aggregated_Executor_Modes::EAGER) {
        // TODO get future and add continuation for when the stream does its
        // thing
        // auto fut = dummy_stream_promise.get_future();
        auto fut = executor.get_future();
        current_continuation = fut.then([this](auto &&fut) {
          std::lock_guard<std::mutex> guard(mut);
          if (!slices_exhausted) {
            slices_exhausted = true;
            launched_slices = current_slices;
            size_t id = 0;
            for (auto &slice_promise : executor_slices) {
              slice_promise.set_value(
                  Executor_Slice{*this, id, current_slices});
              id++;
            }
            executor_slices.clear();
          }
        });
      }
      if (current_slices >= max_slices &&
          mode != Aggregated_Executor_Modes::ENDLESS) {
        slices_exhausted = true;
        launched_slices = current_slices;
        size_t id = 0;
        for (auto &slice_promise : executor_slices) {
          slice_promise.set_value(Executor_Slice{*this, id, current_slices});
          id++;
        }
        executor_slices.clear();
      }
      return ret_fut;
    } else {
      // Return empty optional as failure
      return std::optional<hpx::lcos::future<Executor_Slice>>{};
    }
  }
  size_t launched_slices;
  void reduce_usage_counter(void) {
    std::lock_guard<std::mutex> guard(mut);
    assert(slices_exhausted);
    assert(current_slices >= 0 && current_slices <= launched_slices);
    // First slice goes out of scope?
    if (current_slices == launched_slices) {
      // Finish the continuation to not leave a dangling task!
      // otherwise the continuation might access data of non-existent object...
      current_continuation.get();
      last_stream_launch_done.get();
    }
    current_slices--;
    // Last slice goes out scope?
    if (current_slices == 0) {
      std::lock_guard<std::mutex> guard(buffer_mut);
      function_calls.clear();
#ifndef NDEBUG
      for (const auto &buffer_entry : buffer_allocations) {
        const auto &[buffer_pointer_any, buffer_size, buffer_allocation_counter,
                     valid] = buffer_entry;
        assert(!valid);
      }
#endif
      buffer_allocations.clear();
      // Mark executor fit for reusage
      slices_exhausted = false;
    }
  }
  ~Aggregated_Executor(void) {
    // Aggregated exector should only be deleted if there's currently no aggregation calls going on
    assert(function_calls.empty());
    assert(buffer_allocations.empty());
    assert(current_slices == 0);
    assert(!slices_exhausted);
  }

  // TODO Change executor constructor to query stream manager
  Aggregated_Executor(const size_t number_slices,
                      Aggregated_Executor_Modes mode)
      : max_slices(number_slices), current_slices(0), slices_exhausted(false),
        mode(mode),
        executor(std::get<0>(
            stream_pool::get_interface<Dummy_Executor,
                                       round_robin_pool<Dummy_Executor>>())),
        current_continuation(hpx::lcos::make_ready_future()),
        last_stream_launch_done(hpx::lcos::make_ready_future()) {}
  // Not meant to be copied or moved
  Aggregated_Executor(const Aggregated_Executor &other) = delete;
  Aggregated_Executor &operator=(const Aggregated_Executor &other) = delete;
  Aggregated_Executor(Aggregated_Executor &&other) = delete;
  Aggregated_Executor &operator=(Aggregated_Executor &&other) = delete;
};

template <typename T, typename Host_Allocator, typename Executor>
class Allocator_Slice {
private:
  typename Aggregated_Executor<Executor>::Executor_Slice &executor_reference;

public:
  using value_type = T;
  Allocator_Slice(
      typename Aggregated_Executor<Executor>::Executor_Slice &executor)
      : executor_reference(executor) {}
  template <typename U>
  explicit Allocator_Slice(
      Allocator_Slice<U, Host_Allocator, Executor> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data = executor_reference.template get<T, Host_Allocator>(n);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    executor_reference.template mark_unused<T, Host_Allocator>(p, n);
  }
  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    // Do nothing here - we reuse the content of the last owner
  }
  void destroy(T *p) {
    // Do nothing here - Contents will be destroyed when the buffer manager is
    // destroyed, not before
  }
  // TODO Reference counting not supported yet
  /*void increase_usage_counter(T *p, size_t n) {
    buffer_recycler::increase_usage_counter<T, Host_Allocator>(p, n);
  }*/
};
template <typename T, typename U, typename Host_Allocator, typename Executor>
constexpr bool
operator==(Allocator_Slice<T, Host_Allocator, Executor> const &,
           Allocator_Slice<U, Host_Allocator, Executor> const &) noexcept {
  return true;
}
template <typename T, typename U, typename Host_Allocator, typename Executor>
constexpr bool
operator!=(Allocator_Slice<T, Host_Allocator, Executor> const &,
           Allocator_Slice<U, Host_Allocator, Executor> const &) noexcept {
  return false;
}

//===============================================================================
//===============================================================================
// Pool Strategy:

template <const char *kernelname, class Interface, class Pool>
class aggregation_pool {
public:
  /// interface
  template <typename... Ts>
  static void init(size_t number_of_executors, size_t slices_per_executor,
                   Aggregated_Executor_Modes mode) {
    std::lock_guard<std::mutex> guard(instance.pool_mutex);
    assert(instance.aggregation_executor_pool.empty());
    for (int i = 0; i < number_of_executors; i++) {
      instance.aggregation_executor_pool.emplace_back(slices_per_executor,
                                                      mode);
    }
    instance.slices_per_executor = slices_per_executor;
    instance.mode = mode;
  }

  /// Will always return a valid executor slice
  static decltype(auto) request_executor_slice(void) {
    assert(!instance.aggregation_executor_pool.empty());
    std::optional<hpx::lcos::future<
        typename Aggregated_Executor<Interface>::Executor_Slice>>
        ret;
    size_t local_id = (instance.current_interface) %
                      instance.aggregation_executor_pool.size();
    ret = instance.aggregation_executor_pool[local_id].request_executor_slice();
    // Expected case: current aggregation executor is free
    if (ret.has_value()) {
      return ret;
    }
    // Find next free aggregated_executor
    std::lock_guard<std::mutex>(instance.pool_mutex);
    // retest after lock
    local_id = (instance.current_interface) %
               instance.aggregation_executor_pool.size();
    ret = instance.aggregation_executor_pool[local_id].request_executor_slice();
    if (ret.has_value()) {
      return ret;
    }
    // current interface is still bad -> find free one
    size_t abort_counter = 0;
    const size_t abort_number = instance.aggregation_executor_pool.size() + 1;
    do {
      local_id = (++(instance.current_interface)) % // increment interface
                 instance.aggregation_executor_pool.size();
      ret =
          instance.aggregation_executor_pool[local_id].request_executor_slice();
      if (ret.has_value()) {
        return ret;
      }
      abort_counter++;
    } while (abort_counter <= abort_number);
    // Everything's busy -> create new aggregation executor (growing pool) OR
    // return empty optional
    if (instance.growing_pool) {
      instance.aggregation_executor_pool.emplace_back(
          instance.slices_per_executor, instance.mode);
      ret = instance.aggregation_executor_pool.back().request_executor_slice();
      assert(ret.has_value()); // fresh executor -- should always have slices
                               // available
    }
    return ret;
  }

private:
  std::deque<Aggregated_Executor<Interface>> aggregation_executor_pool;
  std::atomic<size_t> current_interface{0};
  size_t slices_per_executor;
  Aggregated_Executor_Modes mode;
  bool growing_pool{true};

private:
  /// Required for dealing with adding elements to the deque of
  /// aggregated_executors
  static inline std::mutex pool_mutex;
  /// Global access instance
  static inline aggregation_pool instance{};
  aggregation_pool() = default;

public:
  ~aggregation_pool() = default;
  // Bunch of constructors we don't need
  aggregation_pool(aggregation_pool const &other) = delete;
  aggregation_pool &operator=(aggregation_pool const &other) = delete;
  aggregation_pool(aggregation_pool &&other) = delete;
  aggregation_pool &operator=(aggregation_pool &&other) = delete;
};

/*template <class Interface> class aggregation_pool {
private:
  std::deque<Interface> pool{};
  std::vector<size_t> ref_counters{};
  size_t current_interface{0};

  const size_t slices;

public:
  template <typename... Ts>
  explicit aggregation_pool(size_t number_of_streams, size_t slices)
      : slices(slices) {
    ref_counters.reserve(number_of_streams);
    for (int i = 0; i < number_of_streams; i++) {
      pool.emplace_back(slices, Aggregated_Executor_Modes::EAGER);
      ref_counters.emplace_back(0);
    }
  }
  // return a tuple with the interface and its index (to release it later)
  std::tuple<Interface &, size_t> get_interface() {
    if (pool[current_interface].slice_available()) {
      ref_counters[current_interface]++;
      std::tuple<Interface &, size_t> ret(pool[current_interface],
                                          current_interface);
      return ret;
    }
    size_t counter = 0;
    do {
      current_interface = (current_interface + 1) % pool.size();
      counter++;
    } while (!pool[current_interface].slice_available() &&
             counter < pool.size());

    if (pool[current_interface].slice_available()) {
      ref_counters[current_interface]++;
      std::tuple<Interface &, size_t> ret(pool[current_interface],
                                          current_interface);
      return ret;
    }
    pool.emplace_back(slices, Aggregated_Executor_Modes::EAGER);
    ref_counters.emplace_back(0);

    current_interface = pool.size() - 1;
    ref_counters[current_interface]++;
    std::tuple<Interface &, size_t> ret(pool[current_interface],
                                        current_interface);
    return ret;
  }
  void release_interface(size_t index) { ref_counters[index]--; }
  bool interface_available(size_t load_limit) {
    return *(std::min_element(std::begin(ref_counters),
                              std::end(ref_counters))) < load_limit;
  }
  size_t get_current_load() {
    return *(
        std::min_element(std::begin(ref_counters), std::end(ref_counters)));
  }
  size_t get_next_device_id() {
    return 0; // single gpu pool
  }
};*/

//===============================================================================
//===============================================================================
// Test scenarios
//

void sequential_test(void) {
  static const char kernelname[] = "kernel1";
  using kernel_pool1 = aggregation_pool<kernelname, Dummy_Executor,
                                        round_robin_pool<Dummy_Executor>>;
  kernel_pool1::init(8, 2, Aggregated_Executor_Modes::STRICT);
  // Sequential test
  hpx::cout << "Sequential test with all executor slices" << std::endl;
  hpx::cout << "----------------------------------------" << std::endl;
  {
    // Aggregated_Executor<decltype(executor1)> agg_exec{
    //   4, Aggregated_Executor_Modes::STRICT};*/
    auto &agg_exec =
        std::get<0>(stream_pool::get_interface<
                    Aggregated_Executor<Dummy_Executor>,
                    round_robin_pool<Aggregated_Executor<Dummy_Executor>>>());

    std::vector<hpx::lcos::future<void>> slices_done_futs;

    auto slice_fut1 = kernel_pool1::request_executor_slice();
    if (slice_fut1.has_value()) {
      slices_done_futs.emplace_back(slice_fut1.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        auto alloc_int =
            slice_exec.template make_allocator<int, std::allocator<int>>();
        hpx::cout << "Executor 1 ID is " << slice_exec.id << std::endl;
        std::vector<float, decltype(alloc)> some_data(
            slice_exec.number_slices * 10, float{}, alloc);
        std::vector<float, decltype(alloc)> some_data2(
            slice_exec.number_slices * 20, float{}, alloc);
        std::vector<int, decltype(alloc_int)> some_ints(
            slice_exec.number_slices * 20, int{}, alloc_int);
        std::vector<float, decltype(alloc)> some_vector(
            slice_exec.number_slices * 10, float{}, alloc);
        hpx::cout << "Executor 1 Data address is " << some_data.data()
                  << std::endl;

        int i = 1;
        float j = 2;
        slice_exec.post(print_stuff1, i);
        slice_exec.post(print_stuff2, i, j);
        auto kernel_fut = slice_exec.async(print_stuff1, i);
        kernel_fut.get();
      }));
    } else {
      hpx::cerr << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut2 = kernel_pool1::request_executor_slice();
    if (slice_fut2.has_value()) {
      slices_done_futs.emplace_back(slice_fut2.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        auto alloc_int =
            slice_exec.template make_allocator<int, std::allocator<int>>();
        hpx::cout << "Executor 2 ID is " << slice_exec.id << std::endl;
        std::vector<float, decltype(alloc)> some_data(
            slice_exec.number_slices * 10, float{}, alloc);
        std::vector<float, decltype(alloc)> some_data2(
            slice_exec.number_slices * 20, float{}, alloc);
        std::vector<int, decltype(alloc_int)> some_ints(
            slice_exec.number_slices * 20, int{}, alloc_int);
        std::vector<float, decltype(alloc)> some_vector(
            slice_exec.number_slices * 10, float{}, alloc);
        hpx::cout << "Executor 2 Data address is " << some_data.data()
                  << std::endl;
        int i = 1;
        float j = 2;
        slice_exec.post(print_stuff1, i);
        slice_exec.post(print_stuff2, i, j);
        auto kernel_fut = slice_exec.async(print_stuff1, i);
        kernel_fut.get();
      }));
    } else {
      hpx::cerr << "ERROR: Slice 2 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 2 was not created properly");
    }

    auto slice_fut3 = kernel_pool1::request_executor_slice();
    if (slice_fut3.has_value()) {
      slices_done_futs.emplace_back(slice_fut3.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        auto alloc_int =
            slice_exec.template make_allocator<int, std::allocator<int>>();
        hpx::cout << "Executor 3 ID is " << slice_exec.id << std::endl;
        std::vector<float, decltype(alloc)> some_data(
            slice_exec.number_slices * 10, float{}, alloc);
        std::vector<float, decltype(alloc)> somedata2(
            slice_exec.number_slices * 20, float{}, alloc);
        std::vector<int, decltype(alloc_int)> some_ints(
            slice_exec.number_slices * 20, int{}, alloc_int);
        std::vector<float, decltype(alloc)> some_vector(
            slice_exec.number_slices * 10, float{}, alloc);
        hpx::cout << "Executor 3 Data address is " << some_data.data()
                  << std::endl;
        int i = 1;
        float j = 2;
        slice_exec.post(print_stuff1, i);
        slice_exec.post(print_stuff2, i, j);
        auto kernel_fut = slice_exec.async(print_stuff1, i);
        kernel_fut.get();
      }));
    } else {
      hpx::cerr << "ERROR: Slice 3 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 3 was not created properly");
    }

    auto slice_fut4 = kernel_pool1::request_executor_slice();
    if (slice_fut4.has_value()) {
      slices_done_futs.emplace_back(slice_fut4.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        auto alloc_int =
            slice_exec.template make_allocator<int, std::allocator<int>>();
        hpx::cout << "Executor 4 ID is " << slice_exec.id << std::endl;
        std::vector<float, decltype(alloc)> some_data(
            slice_exec.number_slices * 10, float{}, alloc);
        std::vector<float, decltype(alloc)> some_data2(
            slice_exec.number_slices * 20, float{}, alloc);
        std::vector<int, decltype(alloc_int)> some_ints(
            slice_exec.number_slices * 20, int{}, alloc_int);
        std::vector<float, decltype(alloc)> some_vector(
            slice_exec.number_slices * 10, float{}, alloc);
        hpx::cout << "Executor 4 Data address is " << some_data.data()
                  << std::endl;
        int i = 1;
        float j = 2;
        slice_exec.post(print_stuff1, i);
        slice_exec.post(print_stuff2, i, j);
        auto kernel_fut = slice_exec.async(print_stuff1, i);
        kernel_fut.get();
      }));
    } else {
      hpx::cerr << "ERROR: Slice 4 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 4 was not created properly");
    }
    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by equesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
  }
  hpx::cout << std::endl;
}

void interruption_test(void) {
  // Interruption test
  hpx::cout << "Sequential test with interruption:" << std::endl;
  hpx::cout << "----------------------------------" << std::endl;
  {
    Aggregated_Executor<Dummy_Executor> agg_exec{
        4, Aggregated_Executor_Modes::EAGER};
    std::vector<hpx::lcos::future<void>> slices_done_futs;

    auto slice_fut1 = agg_exec.request_executor_slice();
    if (slice_fut1.has_value()) {
      slices_done_futs.emplace_back(slice_fut1.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        hpx::cout << "Got executor 1" << std::endl;
        slice_exec.post(print_stuff1, 1);
        slice_exec.post(print_stuff2, 1, 2.0);
        auto kernel_fut = slice_exec.async(print_stuff1, 1);
        kernel_fut.get();
      }));
    } else {
      hpx::cerr << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    /*auto slice_fut2 = agg_exec.request_executor_slice();
    if (slice_fut2.has_value()) {
      slices_done_futs.emplace_back(slice_fut2.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        hpx::cout << "Got executor 2" << std::endl;
        slice_exec.post(print_stuff1, 1);
        slice_exec.post(print_stuff2, 1, 2.0);
        auto kernel_fut = slice_exec.async(print_stuff1, 1);
        kernel_fut.get();
      }));
    } else {
      hpx::cerr << "ERROR: Slice 2 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 2 was not created properly");
    }

    auto slice_fut3 = agg_exec.request_executor_slice();
    if (slice_fut3.has_value()) {
      slices_done_futs.emplace_back(slice_fut3.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        hpx::cout << "Got executor 3" << std::endl;
        slice_exec.post(print_stuff1, 1);
        slice_exec.post(print_stuff2, 1, 2.0);
        auto kernel_fut = slice_exec.async(print_stuff1, 1);
        kernel_fut.get();
      }));
    } else {
      hpx::cerr << "ERROR: Slice 3 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 3 was not created properly");
    }*/

    hpx::cout << "Requested 1 executors!" << std::endl;
    hpx::cout << "Realizing by setting the continuation future..." << std::endl;
    // Interrupt - should cause executor to start executing all slices
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
  }
  hpx::cout << std::endl;
  // recycler::force_cleanup();
}

void failure_test(void) {
  // Error test
  hpx::cout << "Error test with all wrong types and values in 2 slices"
            << std::endl;
  hpx::cout << "------------------------------------------------------"
            << std::endl;
  {
    Aggregated_Executor<Dummy_Executor> agg_exec{4, Aggregated_Executor_Modes::STRICT};

    auto slice_fut1 = agg_exec.request_executor_slice();

    std::vector<hpx::lcos::future<void>> slices_done_futs;
    slices_done_futs.emplace_back(slice_fut1.value().then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 1" << std::endl;
      slice_exec.post(print_stuff1, 3);
    }));

    auto slice_fut2 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut2.value().then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 2" << std::endl;
      slice_exec.post(print_stuff1, 3);
    }));

    auto slice_fut3 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut3.value().then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 3" << std::endl;
      try {
        slice_exec.post(print_stuff1, 3.0f);
      } catch (...) {
        hpx::cerr << "TEST succeeded: Found type error exception!\n"
                  << std::endl;
        throw;
      }
    }));

    auto slice_fut4 = agg_exec.request_executor_slice();
    slices_done_futs.emplace_back(slice_fut4.value().then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 4" << std::endl;
      // TODO How to propagate the exception?
      try {
        slice_exec.post(print_stuff_error, 3);
      } catch (...) {
        hpx::cerr << "TEST succeeded: Found value error exception!\n"
                  << std::endl;
        throw;
      }
    }));

    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by equesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
  }
  hpx::cout << std::endl;
}

void pointer_add_test(void) {
  hpx::cout << "Host aggregated add pointer example (no references used)"
            << std::endl;
  hpx::cout << "--------------------------------------------------------"
            << std::endl;
  static const char kernelname2[] = "kernel2";
  using kernel_pool2 = aggregation_pool<kernelname2, Dummy_Executor,
                                        round_robin_pool<Dummy_Executor>>;
  kernel_pool2::init(8, 2, Aggregated_Executor_Modes::STRICT);
  {
    std::vector<float> erg(512);
    std::vector<hpx::lcos::future<void>> slices_done_futs;

    auto slice_fut1 = kernel_pool2::request_executor_slice();

    if (slice_fut1.has_value()) {
      slices_done_futs.emplace_back(slice_fut1.value().then([&erg](auto &&fut) {
        // Get slice executor
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        size_t chunksize = 512 / slice_exec.number_slices;
        const size_t task_id = 0;
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(128 * slice_exec.number_slices,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = task_id + 1;
          B[i] = 2 * task_id;
        }

        // Run add function
        auto kernel_fut =
            slice_exec.async(add_pointer<float>, slice_exec.number_slices * 128,
                             A.data(), B.data(), C.data());
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = task_id * 128, j = slice_exec.id * 128;
             i < (task_id + 1) * 128; i++, j++) {
          erg[i] = C[j];
        }
      }));
    } else {
      hpx::cerr << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut2 = kernel_pool2::request_executor_slice();
    if (slice_fut2.has_value()) {
      slices_done_futs.emplace_back(slice_fut2.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        size_t chunksize = 512 / slice_exec.number_slices;
        const size_t task_id = 1;
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(128 * slice_exec.number_slices,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = task_id + 1;
          B[i] = 2 * task_id;
        }

        // Run add function
        auto kernel_fut =
            slice_exec.async(add_pointer<float>, slice_exec.number_slices * 128,
                             A.data(), B.data(), C.data());
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = task_id * 128, j = slice_exec.id * 128;
             i < (task_id + 1) * 128; i++, j++) {
          erg[i] = C[j];
        }
      }));
    } else {
      hpx::cerr << "ERROR: Slice 2 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 2 was not created properly");
    }

    auto slice_fut3 = kernel_pool2::request_executor_slice();
    if (slice_fut3.has_value()) {
      slices_done_futs.emplace_back(slice_fut3.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        size_t chunksize = 512 / slice_exec.number_slices;
        const size_t task_id = 2;
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(128 * slice_exec.number_slices,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = task_id + 1;
          B[i] = 2 * task_id;
        }

        // Run add function
        auto kernel_fut =
            slice_exec.async(add_pointer<float>, slice_exec.number_slices * 128,
                             A.data(), B.data(), C.data());
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = task_id * 128, j = slice_exec.id * 128;
             i < (task_id + 1) * 128; i++, j++) {
          erg[i] = C[j];
        }
      }));
    } else {
      hpx::cerr << "ERROR: Slice 3 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 3 was not created properly");
    }

    auto slice_fut4 = kernel_pool2::request_executor_slice();
    if (slice_fut4.has_value()) {
      slices_done_futs.emplace_back(slice_fut4.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        size_t chunksize = 512 / slice_exec.number_slices;
        const size_t task_id = 3;
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(128 * slice_exec.number_slices,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = task_id + 1;
          B[i] = 2 * task_id;
        }

        // Run add function
        auto kernel_fut =
            slice_exec.async(add_pointer<float>, slice_exec.number_slices * 128,
                             A.data(), B.data(), C.data());
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = task_id * 128, j = slice_exec.id * 128;
             i < (task_id + 1) * 128; i++, j++) {
          erg[i] = C[j];
        }
      }));
    } else {
      hpx::cerr << "ERROR: Slice 4 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 4 was not created properly");
    }
    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by requesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();

    hpx::cout << "Number add_pointer_launches=" << add_pointer_launches
              << std::endl;
    assert(add_pointer_launches == 2);
    hpx::cout << "Checking erg: " << std::endl;
    for (int slice = 0; slice < 4; slice++) {
      for (int i = slice * 128; i < (slice + 1) * 128; i++) {
        assert(erg[i] == 3 * slice + 1);
        hpx::cout << erg[i] << " ";
      }
    }
    hpx::cout << std::endl;
  }
  // recycler::force_cleanup();
  hpx::cout << std::endl;
}

void references_add_test(void) {
  hpx::cout << "Host aggregated add vector example (references used)"
            << std::endl;
  hpx::cout << "----------------------------------------------------"
            << std::endl;
  {
    /*Aggregated_Executor<decltype(executor1)> agg_exec{
        4, Aggregated_Executor_Modes::STRICT};*/
    auto &agg_exec =
        std::get<0>(stream_pool::get_interface<
                    Aggregated_Executor<Dummy_Executor>,
                    round_robin_pool<Aggregated_Executor<Dummy_Executor>>>());
    std::vector<float> erg(512);
    std::vector<hpx::lcos::future<void>> slices_done_futs;

    auto slice_fut1 = agg_exec.request_executor_slice();
    if (slice_fut1.has_value()) {
      slices_done_futs.emplace_back(slice_fut1.value().then([&erg](auto &&fut) {
        // Get slice executor
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(slice_exec.number_slices * 128,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = slice_exec.id + 1;
          B[i] = 2 * slice_exec.id;
        }

        // Run add function
        auto kernel_fut = slice_exec.async(add<decltype(A)>, 128, A, B, C);
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          erg[i] = C[i];
        }
      }));
    } else {
      hpx::cerr << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut2 = agg_exec.request_executor_slice();
    if (slice_fut2.has_value()) {
      slices_done_futs.emplace_back(slice_fut2.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(slice_exec.number_slices * 128,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = slice_exec.id + 1;
          B[i] = 2 * slice_exec.id;
        }

        // Run add function
        auto kernel_fut = slice_exec.async(add<decltype(A)>, 128, A, B, C);
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          erg[i] = C[i];
        }
      }));
    } else {
      hpx::cerr << "ERROR: Slice 2 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 2 was not created properly");
    }

    auto slice_fut3 = agg_exec.request_executor_slice();
    if (slice_fut3.has_value()) {
      slices_done_futs.emplace_back(slice_fut3.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(slice_exec.number_slices * 128,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = slice_exec.id + 1;
          B[i] = 2 * slice_exec.id;
        }

        // Run add function
        auto kernel_fut = slice_exec.async(add<decltype(A)>, 128, A, B, C);
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          erg[i] = C[i];
        }
      }));
    } else {
      hpx::cerr << "ERROR: Slice 3 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 3 was not created properly");
    }

    auto slice_fut4 = agg_exec.request_executor_slice();
    if (slice_fut4.has_value()) {
      slices_done_futs.emplace_back(slice_fut4.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(slice_exec.number_slices * 128,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = slice_exec.id + 1;
          B[i] = 2 * slice_exec.id;
        }

        // Run add function
        auto kernel_fut = slice_exec.async(add<decltype(A)>, 128, A, B, C);
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          erg[i] = C[i];
        }
      }));
    } else {
      hpx::cerr << "ERROR: Slice 4 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 4 was not created properly");
    }
    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by requesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();

    hpx::cout << "Checking erg: " << std::endl;
    for (int slice = 0; slice < 4; slice++) {
      for (int i = slice * 128; i < (slice + 1) * 128; i++) {
        assert(erg[i] == 3 * slice + 1);
        hpx::cout << erg[i] << " ";
      }
    }
    hpx::cout << std::endl;
  }
  hpx::cout << std::endl;

  hpx::cout << "Done!" << std::endl;
  hpx::cout << std::endl;
}

//===============================================================================
//===============================================================================
int hpx_main(int argc, char *argv[]) {
  // Init parameters
  std::string scenario{};
  std::string filename{};
  {
    try {
    boost::program_options::options_description desc{"Options"};
    desc.add_options()("help", "Help screen")(
        "scenario",
        boost::program_options::value<std::string>(&scenario)->default_value(
            "all"),
        "Which scenario to run [sequential_test, interruption_test, failure_test, pointer_add_test, references_add_test, all]")(
        "outputfile",
        boost::program_options::value<std::string>(&filename)->default_value(
            ""),
        "Redirect stdout/stderr to this file");

      boost::program_options::variables_map vm;
      boost::program_options::parsed_options options =
          parse_command_line(argc, argv, desc);
      boost::program_options::store(options, vm);
      boost::program_options::notify(vm);

      if (vm.count("help") == 0u) {
        hpx::cout << "Running with parameters:" << std::endl 
          << "--scenario=" << scenario << std::endl
          << "--outputfile=" << filename << std::endl;
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
  if (scenario != "sequential_test" && scenario != "interruption_test" && scenario != "failure_test" && scenario != "pointer_add_test" && scenario != "references_add_test" && scenario != "all") {
    hpx::cerr << "ERROR: Invalid scenario specified (see --help)" << std::endl;
    return hpx::finalize();
  }

  stream_pool::init<hpx::cuda::experimental::cuda_executor,
                    round_robin_pool<hpx::cuda::experimental::cuda_executor>>(
      8, 0, false);
  stream_pool::init<Dummy_Executor, round_robin_pool<Dummy_Executor>>(8);

  stream_pool::init<Aggregated_Executor<Dummy_Executor>,
                    round_robin_pool<Aggregated_Executor<Dummy_Executor>>>(
      8, 4, Aggregated_Executor_Modes::STRICT);
  /*hpx::cuda::experimental::cuda_executor executor1 =
      std::get<0>(stream_pool::get_interface<
                  hpx::cuda::experimental::cuda_executor,
                  round_robin_pool<hpx::cuda::experimental::cuda_executor>>());*/

  // Basic tests:
  if (scenario == "sequential_test" || scenario == "all" ) {
    sequential_test();
  }
  if (scenario == "interruption_test" || scenario == "all" ) {
    interruption_test();
  }
  if (scenario == "pointer_add_test" || scenario == "all" ) {
    pointer_add_test();
  }
  if (scenario == "references_add_test" || scenario == "all" ) {
    references_add_test();
  }
  // Test that checks failure detection in case of wrong usage (missmatching calls/types/values)
  if (scenario == "failure_test" ) {
    failure_test();
  }

  recycler::force_cleanup(); // Cleanup all buffers and the managers 
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
