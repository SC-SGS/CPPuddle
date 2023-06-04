#ifndef WORK_AGGREGATION_MANAGER
#define WORK_AGGREGATION_MANAGER

#define DEBUG_AGGREGATION_CALLS 1

#include <stdio.h>

#include <any>
#include <atomic>
#include <chrono>
#include <cstdio>
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
#include <unordered_map>

#include <hpx/futures/future.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/promise.hpp>
//#include <hpx/synchronization/mutex.hpp> // obsolete
#include <hpx/mutex.hpp>

#if defined(HPX_HAVE_CUDA) || defined(HPX_HAVE_HIP)
// required for defining type traits using cuda executor as underlying
// aggregation executors
#include <hpx/async_cuda/cuda_executor.hpp>
#endif

#include <boost/core/demangle.hpp>
#include <boost/format.hpp>

#include "../include/buffer_manager.hpp"
#include "../include/stream_manager.hpp"

using aggregation_mutex_t = hpx::mutex;

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

//===============================================================================
//===============================================================================
template <typename Executor, typename F, typename... Ts>
void exec_post_wrapper(Executor & exec, F &&f, Ts &&...ts) {
  hpx::apply(exec, std::forward<F>(f), std::forward<Ts>(ts)...);
}

template <typename Executor, typename F, typename... Ts>
hpx::lcos::future<void> exec_async_wrapper(Executor & exec, F &&f, Ts &&...ts) {
  return hpx::async(exec, std::forward<F>(f), std::forward<Ts>(ts)...);
}

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
  /* hpx::lcos::local::promise<void> slices_ready_promise; */
  /// Tracks if all slices have visited this function call
  /* hpx::lcos::future<void> all_slices_ready = slices_ready_promise.get_future(); */
  /// How many slices can we expect?
  const size_t number_slices;
  const bool async_mode;

  Executor &underlying_executor;

#if !(defined(NDEBUG)) && defined(DEBUG_AGGREGATION_CALLS)
#pragma message                                                                \
    "Running slow work aggegator debug build! Run with NDEBUG defined for fast build..."
  /// Stores the function call of the first slice as reference for error
  /// checking
  std::any function_tuple;
  /// Stores the string of the first function call for debug output
  std::string debug_type_information;
  aggregation_mutex_t debug_mut;
#endif

  std::vector<hpx::lcos::local::promise<void>> potential_async_promises{};

public:
  aggregated_function_call(const size_t number_slices, bool async_mode, Executor &exec)
      : number_slices(number_slices), async_mode(async_mode), underlying_executor(exec) {
    if (async_mode)
      potential_async_promises.resize(number_slices);
  }
  ~aggregated_function_call(void) {
    // All slices should have done this call
    assert(slice_counter == number_slices);
    // assert(!all_slices_ready.valid());
  }
  /// Returns true if all required slices have visited this point
  bool sync_aggregation_slices(hpx::lcos::future<void> &stream_future) {
    assert(!async_mode);
    assert(potential_async_promises.empty());
    const size_t local_counter = slice_counter++;
    if (local_counter == number_slices - 1) {
      return true;
    }
    else return false;
  }
  template <typename F, typename... Ts>
  void post_when(hpx::lcos::future<void> &stream_future, F &&f, Ts &&...ts) {
#if !(defined(NDEBUG)) && defined(DEBUG_AGGREGATION_CALLS)
    // needed for concurrent access to function_tuple and debug_type_information
    // Not required for normal use
    std::lock_guard<aggregation_mutex_t> guard(debug_mut);
#endif
    assert(!async_mode);
    assert(potential_async_promises.empty());
    const size_t local_counter = slice_counter++;

    if (local_counter == 0) {
#if !(defined(NDEBUG)) && defined(DEBUG_AGGREGATION_CALLS)
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
#if !(defined(NDEBUG)) && defined(DEBUG_AGGREGATION_CALLS)
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
        hpx::cout
            << "\nMismatched types error in aggregated post call of executor "
            << ": " << e.what() << "\n";
        hpx::cout << "Expected types:\t\t "
                  << boost::core::demangle(debug_type_information.c_str());
        hpx::cout << "\nGot types:\t\t "
                  << boost::core::demangle(
                         typeid(decltype(comparison_tuple)).name())
                  << "\n"
                  << std::endl;
        // throw;
      } catch (const std::runtime_error &e) {
        hpx::cout
            << "\nMismatched values error in aggregated post call of executor "
            << ": " << e.what() << std::endl;
        hpx::cout << "Types (matched):\t "
                  << boost::core::demangle(debug_type_information.c_str());
        auto orig_call_tuple =
            std::any_cast<decltype(comparison_tuple)>(function_tuple);
        hpx::cout << "\nExpected values:\t ";
        print_tuple(orig_call_tuple);
        hpx::cout << "\nGot values:\t\t ";
        print_tuple(comparison_tuple);
        hpx::cout << std::endl << std::endl;
        // throw;
      }
#endif
    }
    assert(local_counter < number_slices);
    assert(slice_counter < number_slices + 1);
    // Check exit criteria: Launch function call continuation by setting the
    // slices promise
    if (local_counter == number_slices - 1) {
      exec_post_wrapper<Executor, F, Ts...>(underlying_executor, std::forward<F>(f), std::forward<Ts>(ts)...);
      //slices_ready_promise.set_value();
    }
  }
  template <typename F, typename... Ts>
  hpx::lcos::future<void> async_when(hpx::lcos::future<void> &stream_future,
                                     F &&f, Ts &&...ts) {
#if !(defined(NDEBUG)) && defined(DEBUG_AGGREGATION_CALLS)
    // needed for concurrent access to function_tuple and debug_type_information
    // Not required for normal use
    std::lock_guard<aggregation_mutex_t> guard(debug_mut);
#endif
    assert(async_mode);
    assert(!potential_async_promises.empty());
    const size_t local_counter = slice_counter++;
    if (local_counter == 0) {
#if !(defined(NDEBUG)) && defined(DEBUG_AGGREGATION_CALLS)
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
#if !(defined(NDEBUG)) && defined(DEBUG_AGGREGATION_CALLS)
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
        hpx::cout
            << "\nMismatched types error in aggregated async call of executor "
            << ": " << e.what() << "\n";
        hpx::cout << "Expected types:\t\t "
                  << boost::core::demangle(debug_type_information.c_str());
        hpx::cout << "\nGot types:\t\t "
                  << boost::core::demangle(
                         typeid(decltype(comparison_tuple)).name())
                  << "\n"
                  << std::endl;
        // throw;
      } catch (const std::runtime_error &e) {
        hpx::cout
            << "\nMismatched values error in aggregated async call of executor "
            << ": " << e.what() << std::endl;
        hpx::cout << "Types (matched):\t "
                  << boost::core::demangle(debug_type_information.c_str());
        auto orig_call_tuple =
            std::any_cast<decltype(comparison_tuple)>(function_tuple);
        hpx::cout << "\nExpected values:\t ";
        print_tuple(orig_call_tuple);
        hpx::cout << "\nGot values:\t\t ";
        print_tuple(comparison_tuple);
        hpx::cout << std::endl << std::endl;
        // throw;
      }
#endif
    }
    assert(local_counter < number_slices);
    assert(slice_counter < number_slices + 1);
    assert(potential_async_promises.size() == number_slices);
    hpx::lcos::future<void> ret_fut =
        potential_async_promises[local_counter].get_future();
    if (local_counter == number_slices - 1) {
      /* slices_ready_promise.set_value(); */
      auto fut = exec_async_wrapper<Executor, F, Ts...>(underlying_executor, std::forward<F>(f), std::forward<Ts>(ts)...);
      fut.then([this](auto &&fut) {
        for (auto &promise : potential_async_promises) {
          promise.set_value();
        }
      });
    }
    // Check exit criteria: Launch function call continuation by setting the
    // slices promise
    return ret_fut;
  }
  template <typename F, typename... Ts>
  hpx::lcos::shared_future<void> wrap_async(hpx::lcos::future<void> &stream_future,
                                     F &&f, Ts &&...ts) {
    assert(async_mode);
    assert(!potential_async_promises.empty());
    const size_t local_counter = slice_counter++;
    assert(local_counter < number_slices);
    assert(slice_counter < number_slices + 1);
    assert(potential_async_promises.size() == number_slices);
    hpx::lcos::shared_future<void> ret_fut =
        potential_async_promises[local_counter].get_shared_future();
    if (local_counter == number_slices - 1) {
      auto fut = f(std::forward<Ts>(ts)...);
      fut.then([this](auto &&fut) {
        // TODO just use one promise
        for (auto &promise : potential_async_promises) {
          promise.set_value();
        }
      });
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
  std::atomic<bool> slices_exhausted;

  std::atomic<bool> executor_slices_alive;
  std::atomic<bool> buffers_in_use;
  std::atomic<size_t> dealloc_counter;

  const Aggregated_Executor_Modes mode;
  const size_t max_slices;
  std::atomic<size_t> current_slices;
  /// Executor reference and its ID in the exextutor pool
  std::tuple<Executor &, size_t> executor_tuple;
  /// Reference to the executor (presumably residing in the executor pool)
  Executor &executor;

public:
  // Subclasses

  /// Slice class - meant as a scope interface to the aggregated executor
  class Executor_Slice {
  public:
    Aggregated_Executor<Executor> &parent;
  private:
    /// Executor is a slice of this aggregated_executor
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
          number_slices(number_slices), id(slice_id) {
  }
    ~Executor_Slice(void) {
      // Don't notify parent if we moved away from this executor_slice
      if (notify_parent_about_destruction) {
        // Executor should be done by the time of destruction
        // -> check here before notifying parent

        // parent still in execution mode?
        assert(parent.slices_exhausted == true);
        // all kernel launches done?
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
    bool sync_aggregation_slices() {
      assert(parent.slices_exhausted == true);
      auto ret = parent.sync_aggregation_slices(launch_counter);
      launch_counter++;
      return ret;
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

    // OneWay Execution
    template <typename F, typename... Ts>
    friend decltype(auto) tag_invoke(hpx::parallel::execution::post_t,
        Executor_Slice& exec, F&& f, Ts&&... ts)
    {
        return exec.post(std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    // TwoWay Execution
    template <typename F, typename... Ts>
    friend decltype(auto) tag_invoke(
        hpx::parallel::execution::async_execute_t, Executor_Slice& exec,
        F&& f, Ts&&... ts)
    {
        return exec.async(
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    template <typename F, typename... Ts>
    hpx::lcos::shared_future<void> wrap_async(F &&f, Ts &&...ts) {
      // we should only execute function calls once all slices
      // have been given away (-> Executor Slices start)
      assert(parent.slices_exhausted == true);
      hpx::lcos::shared_future<void> ret_fut = parent.wrap_async(
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
      assert(buffer_counter > 0);
      return aggregated_buffer;
    }

    Executor& get_underlying_executor(void) {
      return parent.executor;
    }
  };

  //===============================================================================

  hpx::lcos::local::promise<void> slices_full_promise;
  /// Promises with the slice executors -- to be set when the starting criteria
  /// is met
  std::vector<hpx::lcos::local::promise<Executor_Slice>> executor_slices;
  /// List of aggregated function calls - function will be launched when all
  /// slices have called it
  std::deque<aggregated_function_call<Executor>> function_calls;
  /// For synchronizing the access to the function calls list
  aggregation_mutex_t mut;

  /// Data entry for a buffer allocation: void* pointer, size_t for
  /// buffer-size, atomic for the slice counter
  using buffer_entry_t =
      std::tuple<void*, const size_t, std::atomic<size_t>, bool, const size_t>;
  /// Keeps track of the aggregated buffer allocations done in all the slices
  std::deque<buffer_entry_t> buffer_allocations;
  /// Map pointer to deque index for fast access in the deallocations
  std::unordered_map<void*,size_t> buffer_allocations_map;
  /// For synchronizing the access to the buffer_allocations
  aggregation_mutex_t buffer_mut;
  std::atomic<size_t> buffer_counter = 0;

  /// Get new buffer OR get buffer already allocated by different slice
  template <typename T, typename Host_Allocator>
  T *get(const size_t size, const size_t slice_alloc_counter) {
    assert(slices_exhausted == true);
    assert(executor_slices_alive == true);
    // Add aggreated buffer entry in case it hasn't happened yet for this call
    // First: Check if it already has happened
    if (buffer_counter <= slice_alloc_counter) {
      // we might be the first! Lock...
      std::lock_guard<aggregation_mutex_t> guard(buffer_mut);
      // ... and recheck
      if (buffer_counter <= slice_alloc_counter) {
        constexpr bool manage_content_lifetime = false;
        buffers_in_use = true;
        // get prefered location: aka the current hpx threads location
        const size_t location_id = hpx::get_worker_thread_num();
        // Get shiny and new buffer that will be shared between all slices
        // Buffer might be recycled from previous allocations by the
        // buffer_recycler...
        T *aggregated_buffer =
            recycler::detail::buffer_recycler::get<T, Host_Allocator>(
                size, manage_content_lifetime, location_id);
        // Create buffer entry for this buffer
        buffer_allocations.emplace_back(static_cast<void *>(aggregated_buffer),
                                        size, 1, true, location_id);

#ifndef NDEBUG
        // if previousely used the buffer should not be in usage anymore
        const auto exists = buffer_allocations_map.count(
            static_cast<void *>(aggregated_buffer));
        if (exists > 0) {
          const auto previous_usage_id =
              buffer_allocations_map[static_cast<void *>(aggregated_buffer)];
          const auto &valid =
              std::get<3>(buffer_allocations[previous_usage_id]);
          assert(!valid);
        }
#endif
        buffer_allocations_map.insert_or_assign(static_cast<void *>(aggregated_buffer),
            buffer_counter);

        assert (buffer_counter == slice_alloc_counter);
        buffer_counter = buffer_allocations.size();

        // Return buffer
        return aggregated_buffer;
      }
    }
    assert(buffers_in_use == true);
    assert(std::get<3>(buffer_allocations[slice_alloc_counter])); // valid
    assert(std::get<2>(buffer_allocations[slice_alloc_counter]) >= 1);

    // Buffer entry should already exist:
    T *aggregated_buffer = static_cast<T *>(
        std::get<0>(buffer_allocations[slice_alloc_counter]));
    // Error handling: Size is wrong?
    assert(size == std::get<1>(buffer_allocations[slice_alloc_counter]));
    // Notify that one more slice has visited this buffer allocation
    std::get<2>(buffer_allocations[slice_alloc_counter])++;
    return aggregated_buffer;
  }

  /// Notify buffer list that one slice is done with the buffer
  template <typename T, typename Host_Allocator>
  void mark_unused(T *p, const size_t size) {
    assert(slices_exhausted == true);

    void *ptr_key = static_cast<void*>(p);
    size_t slice_alloc_counter = buffer_allocations_map[p];

    assert(slice_alloc_counter < buffer_allocations.size());
    /*auto &[buffer_pointer_any, buffer_size, buffer_allocation_counter, valid] =
        buffer_allocations[slice_alloc_counter];*/
    auto buffer_pointer_void = std::get<0>(buffer_allocations[slice_alloc_counter]);
    const auto buffer_size = std::get<1>(buffer_allocations[slice_alloc_counter]);
    auto &buffer_allocation_counter = std::get<2>(buffer_allocations[slice_alloc_counter]);
    auto &valid = std::get<3>(buffer_allocations[slice_alloc_counter]);
    const auto &location_id = std::get<4>(buffer_allocations[slice_alloc_counter]);
    assert(valid);
    T *buffer_pointer = static_cast<T *>(buffer_pointer_void);

    assert(buffer_size == size);
    assert(p == buffer_pointer);
    // assert(buffer_pointer == p || buffer_pointer == nullptr);
    // Slice is done with this buffer
    buffer_allocation_counter--;
    // Check if all slices are done with this buffer?
    if (buffer_allocation_counter == 0) {
      // Yes! "Deallocate" by telling the recylcer the buffer is fit for reusage
      std::lock_guard<aggregation_mutex_t> guard(buffer_mut);
      // Only mark unused if another buffer has not done so already (and marked
      // it as invalid)
      if (valid) {
        assert(buffers_in_use == true);
        recycler::detail::buffer_recycler::mark_unused<T, Host_Allocator>(
            buffer_pointer, buffer_size, location_id);
        // mark buffer as invalid to prevent any other slice from marking the
        // buffer as unused
        valid = false;

        const size_t current_deallocs = ++dealloc_counter;
        if (current_deallocs == buffer_counter) {
          std::lock_guard<aggregation_mutex_t> guard(mut);
          buffers_in_use = false;
          if (!executor_slices_alive && !buffers_in_use)
            slices_exhausted = false;
        }
      }
    }
  }

  //===============================================================================
  // Public Interface
public:
  hpx::lcos::future<void> current_continuation;
  hpx::lcos::future<void> last_stream_launch_done;
  std::atomic<size_t> overall_launch_counter = 0;

  /// Only meant to be accessed by the slice executors
  bool sync_aggregation_slices(const size_t slice_launch_counter) {
    std::lock_guard<aggregation_mutex_t> guard(mut);
    assert(slices_exhausted == true);
    // Add function call object in case it hasn't happened for this launch yet
    if (overall_launch_counter <= slice_launch_counter) {
      /* std::lock_guard<aggregation_mutex_t> guard(mut); */
      if (overall_launch_counter <= slice_launch_counter) {
        function_calls.emplace_back(current_slices, false, executor);
        overall_launch_counter = function_calls.size();
        return function_calls[slice_launch_counter].sync_aggregation_slices(
            last_stream_launch_done);
      }
    }

    return function_calls[slice_launch_counter].sync_aggregation_slices(
        last_stream_launch_done);
  }

  /// Only meant to be accessed by the slice executors
  template <typename F, typename... Ts>
  void post(const size_t slice_launch_counter, F &&f, Ts &&...ts) {
    std::lock_guard<aggregation_mutex_t> guard(mut);
    assert(slices_exhausted == true);
    // Add function call object in case it hasn't happened for this launch yet
    if (overall_launch_counter <= slice_launch_counter) {
      /* std::lock_guard<aggregation_mutex_t> guard(mut); */
      if (overall_launch_counter <= slice_launch_counter) {
        function_calls.emplace_back(current_slices, false, executor);
        overall_launch_counter = function_calls.size();
        function_calls[slice_launch_counter].post_when(
            last_stream_launch_done, std::forward<F>(f), std::forward<Ts>(ts)...);
        return;
      }
    }

    function_calls[slice_launch_counter].post_when(
        last_stream_launch_done, std::forward<F>(f), std::forward<Ts>(ts)...);
    return;
  }

  /// Only meant to be accessed by the slice executors
  template <typename F, typename... Ts>
  hpx::lcos::future<void> async(const size_t slice_launch_counter, F &&f,
                                Ts &&...ts) {
    std::lock_guard<aggregation_mutex_t> guard(mut);
    assert(slices_exhausted == true);
    // Add function call object in case it hasn't happened for this launch yet
    if (overall_launch_counter <= slice_launch_counter) {
      /* std::lock_guard<aggregation_mutex_t> guard(mut); */
      if (overall_launch_counter <= slice_launch_counter) {
        function_calls.emplace_back(current_slices, true, executor);
        overall_launch_counter = function_calls.size();
        return function_calls[slice_launch_counter].async_when(
            last_stream_launch_done, std::forward<F>(f), std::forward<Ts>(ts)...);
      }
    }

    return function_calls[slice_launch_counter].async_when(
        last_stream_launch_done, std::forward<F>(f), std::forward<Ts>(ts)...);
  }
  /// Only meant to be accessed by the slice executors
  template <typename F, typename... Ts>
  hpx::lcos::shared_future<void> wrap_async(const size_t slice_launch_counter, F &&f,
                                Ts &&...ts) {
    std::lock_guard<aggregation_mutex_t> guard(mut);
    assert(slices_exhausted == true);
    // Add function call object in case it hasn't happened for this launch yet
    if (overall_launch_counter <= slice_launch_counter) {
      /* std::lock_guard<aggregation_mutex_t> guard(mut); */
      if (overall_launch_counter <= slice_launch_counter) {
        function_calls.emplace_back(current_slices, true, executor);
        overall_launch_counter = function_calls.size();
        return function_calls[slice_launch_counter].wrap_async(
            last_stream_launch_done, std::forward<F>(f), std::forward<Ts>(ts)...);
      }
    }

    return function_calls[slice_launch_counter].wrap_async(
        last_stream_launch_done, std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  bool slice_available(void) {
    std::lock_guard<aggregation_mutex_t> guard(mut);
    return !slices_exhausted;
  }

  std::optional<hpx::lcos::future<Executor_Slice>> request_executor_slice() {
    std::lock_guard<aggregation_mutex_t> guard(mut);
    if (!slices_exhausted) {
      const size_t local_slice_id = ++current_slices;
      if (local_slice_id == 1) {
        // Cleanup leftovers from last run if any
        // TODO still required? Should be clean here already
        function_calls.clear();
        overall_launch_counter = 0;
        std::lock_guard<aggregation_mutex_t> guard(buffer_mut);
#ifndef NDEBUG
        for (const auto &buffer_entry : buffer_allocations) {
          const auto &[buffer_pointer_any, buffer_size,
                       buffer_allocation_counter, valid, location_id] =
              buffer_entry;
          assert(!valid);
        }
#endif 
        buffer_allocations.clear();
        buffer_allocations_map.clear();
        buffer_counter = 0;

        assert(executor_slices_alive == false);
        assert(buffers_in_use == false);
        executor_slices_alive = true;
        buffers_in_use = false;
        dealloc_counter = 0;

        if (mode == Aggregated_Executor_Modes::STRICT ) {
          slices_full_promise = hpx::lcos::local::promise<void>{};
        }
      }

      // Create Executor Slice future -- that will be returned later
      hpx::lcos::future<Executor_Slice> ret_fut;
      if (local_slice_id < max_slices) {
        executor_slices.emplace_back(hpx::lcos::local::promise<Executor_Slice>{});
        ret_fut =
            executor_slices[local_slice_id - 1].get_future();
      } else {
        launched_slices = current_slices;
        ret_fut = hpx::make_ready_future(Executor_Slice{*this,
            executor_slices.size(), launched_slices});
      }

      // Are we the first slice? If yes, add continuation set the
      // Executor_Slice
      // futures to ready if the launch conditions are met
      if (local_slice_id == 1) {
        // Renew promise that all slices will be ready as the primary launch criteria...
        hpx::lcos::shared_future<void> fut;
        if (mode == Aggregated_Executor_Modes::EAGER || mode == Aggregated_Executor_Modes::ENDLESS) {
          // Fallback launch condidtion: Launch as soon as the underlying stream is ready
          /* auto slices_full_fut = slices_full_promise.get_future(); */
          auto exec_fut = executor.get_future(); 
          /* fut = hpx::when_any(exec_fut, slices_full_fut); */
          fut = std::move(exec_fut);
        } else {
          auto slices_full_fut = slices_full_promise.get_shared_future();
          // Just use the slices launch condition
          fut = std::move(slices_full_fut);
        }
        // Launch all executor slices within this continuation
        current_continuation = fut.then([this](auto &&fut) {
          std::lock_guard<aggregation_mutex_t> guard(mut);
          slices_exhausted = true;
          launched_slices = current_slices;
          size_t id = 0;
          for (auto &slice_promise : executor_slices) {
            slice_promise.set_value(
                Executor_Slice{*this, id, launched_slices});
            id++;
          }
          executor_slices.clear();
        });
      }
      if (local_slice_id >= max_slices &&
          mode != Aggregated_Executor_Modes::ENDLESS) {
        slices_exhausted = true; // prevents any more threads from entering before the continuation is launched
          /* launched_slices = current_slices; */
          /* size_t id = 0; */
          /* for (auto &slice_promise : executor_slices) { */
          /*   slice_promise.set_value( */
          /*       Executor_Slice{*this, id, launched_slices}); */
          /*   id++; */
          /* } */
          /* executor_slices.clear(); */
        if (mode == Aggregated_Executor_Modes::STRICT ) {
          slices_full_promise.set_value(); // Trigger slices launch condition continuation 
        }
        // that continuation will set all executor slices so far handed out to ready
      }
      return ret_fut;
    } else {
      // Return empty optional as failure
      return std::optional<hpx::lcos::future<Executor_Slice>>{};
    }
  }
  size_t launched_slices;
  void reduce_usage_counter(void) {
    /* std::lock_guard<aggregation_mutex_t> guard(mut); */
    assert(slices_exhausted == true);
    assert(executor_slices_alive == true);
    assert(launched_slices >= 1);
    assert(current_slices >= 0 && current_slices <= launched_slices);
    const size_t local_slice_id = --current_slices;
    // Last slice goes out scope?
    if (local_slice_id == 0) {

      // Draw new underlying executor TODO Test if it's better to redraw at
      // the first slice request stream_pool::release_interface<Executor,
      // round_robin_pool<Executor>>( std::get<1>(executor_tuple));
      // executor_tuple = stream_pool::get_interface<Executor,
      // round_robin_pool<Executor>>(); executor =
      // std::get<0>(executor_tuple); 
      // Mark executor fit for reusage

      std::lock_guard<aggregation_mutex_t> guard(mut);
      executor_slices_alive = false; 
      if (!executor_slices_alive && !buffers_in_use) {
        slices_exhausted = false;
      }
    }
  }
  ~Aggregated_Executor(void) {

    assert(current_slices == 0);
    assert(executor_slices_alive == false);
    assert(buffers_in_use == false);

    if (mode != Aggregated_Executor_Modes::STRICT ) {
        slices_full_promise.set_value(); // Trigger slices launch condition continuation 
    }

    // Cleanup leftovers from last run if any
    function_calls.clear();
    overall_launch_counter = 0;
#ifndef NDEBUG
    for (const auto &buffer_entry : buffer_allocations) {
      const auto &[buffer_pointer_any, buffer_size, buffer_allocation_counter,
                   valid, location_id] = buffer_entry;
      assert(!valid);
    }
#endif
    buffer_allocations.clear();
    buffer_allocations_map.clear();
    buffer_counter = 0;

    assert(buffer_allocations.empty());
    assert(buffer_allocations_map.empty());
  }

  Aggregated_Executor(const size_t number_slices,
                      Aggregated_Executor_Modes mode)
      : max_slices(number_slices), current_slices(0), slices_exhausted(false),dealloc_counter(0),
        mode(mode), executor_slices_alive(false), buffers_in_use(false),
        executor_tuple(
            stream_pool::get_interface<Executor, round_robin_pool<Executor>>()),
        executor(std::get<0>(executor_tuple)),
        current_continuation(hpx::make_ready_future()),
        last_stream_launch_done(hpx::make_ready_future()) {}
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
  Aggregated_Executor<Executor> &executor_parent;

public:
  using value_type = T;
  Allocator_Slice(
      typename Aggregated_Executor<Executor>::Executor_Slice &executor)
      : executor_reference(executor), executor_parent(executor.parent) {}
  template <typename U>
  explicit Allocator_Slice(
      Allocator_Slice<U, Host_Allocator, Executor> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data = executor_reference.template get<T, Host_Allocator>(n);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    /* executor_reference.template mark_unused<T, Host_Allocator>(p, n); */
    executor_parent.template mark_unused<T, Host_Allocator>(p, n);
  }
  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    // Do nothing here - we reuse the content of the last owner
  }
  void destroy(T *p) {
    // Do nothing here - Contents will be destroyed when the buffer manager is
    // destroyed, not before
  }
};
template <typename T, typename U, typename Host_Allocator, typename Executor>
constexpr bool
operator==(Allocator_Slice<T, Host_Allocator, Executor> const &,
           Allocator_Slice<U, Host_Allocator, Executor> const &) noexcept {
  return false;
}
template <typename T, typename U, typename Host_Allocator, typename Executor>
constexpr bool
operator!=(Allocator_Slice<T, Host_Allocator, Executor> const &,
           Allocator_Slice<U, Host_Allocator, Executor> const &) noexcept {
  return true;
}

namespace hpx { namespace parallel { namespace execution {
   // TODO Unfortunately does not work that way! Create trait that works for Executor Slices with 
   // compatible unlying executor types
    /* template<typename E> */
    /* struct is_one_way_executor<typename Aggregated_Executor<E>::Executor_Slice> */
    /*   : std::true_type */
    /* {}; */
    /* template<typename E> */
    /* struct is_two_way_executor<typename Aggregated_Executor<E>::Executor_Slice> */
    /*   : std::true_type */
    /* {}; */

#if defined(HPX_HAVE_CUDA) || defined(HPX_HAVE_HIP)
    // Workaround for the meantime: Manually create traits for compatible types:
    template<>
    struct is_one_way_executor<typename Aggregated_Executor<hpx::cuda::experimental::cuda_executor>::Executor_Slice>
      : std::true_type
    {};
    template<>
    struct is_two_way_executor<typename Aggregated_Executor<hpx::cuda::experimental::cuda_executor>::Executor_Slice>
      : std::true_type
    {};
#endif
}}}

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
    std::lock_guard<aggregation_mutex_t> guard(instance.pool_mutex);
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
    std::lock_guard<aggregation_mutex_t> guard(instance.pool_mutex);
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
    // current interface is bad -> find free one
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
      instance.current_interface = instance.aggregation_executor_pool.size() - 1;
      assert(instance.aggregation_executor_pool.size() < 20480);
      ret = instance.aggregation_executor_pool[instance.current_interface].request_executor_slice();
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
  static inline aggregation_mutex_t pool_mutex;
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

#endif
