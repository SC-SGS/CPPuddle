// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXECUTOR_POOLS_MANAGEMENT_HPP
#define EXECUTOR_POOLS_MANAGEMENT_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <tuple>
#include <type_traits>

#include "cppuddle/common/config.hpp"

// Need to cuda/hip definitions for default params when NOT
// drawing from an executor pool
#if defined(CPPUDDLE_DEACTIVATE_EXECUTOR_RECYCLING)
#include <hpx/config.hpp>
#if defined(HPX_HAVE_CUDA) || defined(HPX_HAVE_HIP)
#include <hpx/async_cuda/cuda_executor.hpp>
#endif
#endif

// Redefintion required for non-recycling executors
// Without it, default constructing the executors (independent) would not work
#if defined(CPPUDDLE_DEACTIVATE_EXECUTOR_RECYCLING)
// Do only define if Kokkos is not found
#ifndef KOKKOS_ENABLE_SERIAL
namespace hpx { namespace kokkos {
enum class execution_space_mode { global, independent };
}}
#endif
#endif

namespace cppuddle {
namespace executor_recycling {
namespace detail {

/// Turns a std::array_mutex into an scoped lock
template<typename mutex_array_t>
auto make_scoped_lock_from_array(mutex_array_t& mutexes)
{
    return std::apply([](auto&... mutexes) { return std::scoped_lock{mutexes...}; }, 
                      mutexes);
}

template <typename Interface> class round_robin_pool_impl {
private:
  std::deque<Interface> pool{};
  std::vector<size_t> ref_counters{};
  size_t current_interface{0};

public:
  template <typename... Ts>
  round_robin_pool_impl(size_t number_of_executors, Ts... executor_args) {
    ref_counters.reserve(number_of_executors);
    for (int i = 0; i < number_of_executors; i++) {
      pool.emplace_back(executor_args...);
      ref_counters.emplace_back(0);
    }
  }
  // return a tuple with the interface and its index (to release it later)
  std::tuple<Interface &, size_t> get_interface() {
    assert(!(pool.empty())); 
    size_t last_interface = current_interface;
    current_interface = (current_interface + 1) % pool.size();
    ref_counters[last_interface]++;
    std::tuple<Interface &, size_t> ret(pool[last_interface], last_interface);
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
  // TODO Remove
  /* size_t get_next_device_id() { */
  /*   return 0; // single gpu pool */
  /* } */
};

template <typename Interface> class priority_pool_impl {
private:
  std::deque<Interface> pool{};
  std::vector<size_t> ref_counters{}; // Ref counters
  std::vector<size_t> priorities{};   // Ref counters
public:
  template <typename... Ts>
  priority_pool_impl(size_t number_of_executors, Ts... executor_args) {
    ref_counters.reserve(number_of_executors);
    priorities.reserve(number_of_executors);
    for (auto i = 0; i < number_of_executors; i++) {
      pool.emplace_back(executor_args...);
      ref_counters.emplace_back(0);
      priorities.emplace_back(i);
    }
  }
  // return a tuple with the interface and its index (to release it later)
  std::tuple<Interface &, size_t> get_interface() {
    auto &interface = pool[priorities[0]];
    ref_counters[priorities[0]]++;
    std::tuple<Interface &, size_t> ret(interface, priorities[0]);
    std::make_heap(std::begin(priorities), std::end(priorities),
                   [this](const size_t &first, const size_t &second) -> bool {
                     return ref_counters[first] > ref_counters[second];
                   });
    return ret;
  }
  void release_interface(size_t index) {
    ref_counters[index]--;
    std::make_heap(std::begin(priorities), std::end(priorities),
                   [this](const size_t &first, const size_t &second) -> bool {
                     return ref_counters[first] > ref_counters[second];
                   });
  }
  bool interface_available(size_t load_limit) {
    return ref_counters[priorities[0]] < load_limit;
  }
  size_t get_current_load() { return ref_counters[priorities[0]]; }
  // TODO remove
  /* size_t get_next_device_id() { */
  /*   return 0; // single gpu pool */
  /* } */
};

/// Access/Concurrency Control for executor pool implementation
class executor_pool {
public:
  template <typename Interface, typename Pool, typename... Ts>
  static void init(size_t number_of_executors, Ts ... executor_args) {
    executor_pool_implementation<Interface, Pool>::init(number_of_executors,
                                                      executor_args...);
  }
  template <typename Interface, typename Pool, typename... Ts>
  static void init_all_executor_pools(size_t number_of_executors, Ts ... executor_args) {
    executor_pool_implementation<Interface, Pool>::init_all_executor_pools(number_of_executors,
                                                      executor_args...);
  }
  template <typename Interface, typename Pool, typename... Ts>
  static void init_executor_pool(size_t pool_id, size_t number_of_executors, Ts ... executor_args) {
    executor_pool_implementation<Interface, Pool>::init_executor_pool(pool_id, number_of_executors,
                                                      executor_args...);
  }
  template <typename Interface, typename Pool> static void cleanup() {
    executor_pool_implementation<Interface, Pool>::cleanup();
  }
  template <typename Interface, typename Pool>
  static std::tuple<Interface &, size_t> get_interface(const size_t gpu_id) {
    return executor_pool_implementation<Interface, Pool>::get_interface(gpu_id);
  }
  template <typename Interface, typename Pool>
  static void release_interface(size_t index, const size_t gpu_id) noexcept {
    executor_pool_implementation<Interface, Pool>::release_interface(index,
        gpu_id);
  }
  template <typename Interface, typename Pool>
  static bool interface_available(size_t load_limit, const size_t gpu_id) noexcept {
    return executor_pool_implementation<Interface, Pool>::interface_available(
        load_limit, gpu_id);
  }
  template <typename Interface, typename Pool>
  static size_t get_current_load(const size_t gpu_id = 0) noexcept {
    return executor_pool_implementation<Interface, Pool>::get_current_load(
        gpu_id);
  }
  template <typename Interface, typename Pool>
  static size_t get_next_device_id(const size_t number_gpus) noexcept {
    // TODO add round robin and min strategy
    return cppuddle::get_device_id(number_gpus);
  }

  template <typename Interface, typename Pool>
  static void set_device_selector(std::function<void(size_t)> select_gpu_function) {
    executor_pool_implementation<Interface, Pool>::set_device_selector(select_gpu_function);
  }

  template <typename Interface, typename Pool>
  static void select_device(size_t gpu_id) {
    executor_pool_implementation<Interface, Pool>::select_device(gpu_id);
  }

private:
  executor_pool() = default;

private:
  template <typename Interface, typename Pool> class executor_pool_implementation {
  public:
    /// Deprecated! Use init_on_all_gpu or init_on_gpu
    template <typename... Ts>
    static void init(size_t number_of_executors, Ts ... executor_args) {
      /* static_assert(sizeof...(Ts) == sizeof...(Ts) && cppuddle::max_number_gpus == 1, */
      /*               "deprecated executor_pool::init does not support multigpu"); */
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      instance().executorpools.emplace_back(number_of_executors, executor_args...);
      assert(instance().executorpools.size() <= cppuddle::max_number_gpus);
    }

    /// Multi-GPU init where executors / interfaces on all GPUs are initialized with the same arguments
    template <typename... Ts>
    static void init_all_executor_pools(size_t number_of_executors, Ts ... executor_args) {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      if (number_of_executors > 0) {
        for (size_t gpu_id = 0; gpu_id < cppuddle::max_number_gpus; gpu_id++) {
          instance().select_gpu_function(gpu_id);
          instance().executorpools.emplace_back(number_of_executors,
                                              executor_args...);
        }
      }
      assert(instance().executorpools.size() <= cppuddle::max_number_gpus);
    }

    /// Per-GPU init allowing for different init parameters depending on the GPU 
    /// (useful for executor that expect an GPU-id during construction)
    template <typename... Ts>
    static void init_executor_pool(size_t gpu_id, size_t number_of_executors, Ts ... executor_args) {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      if (number_of_executors > 0) {
        instance().select_gpu_function(gpu_id);
        instance().executorpools.emplace_back(number_of_executors, 
                                            executor_args...);
      }
      assert(instance().executorpools.size() <= cppuddle::max_number_gpus);
    }

    // TODO add/rename into finalize?
    static void cleanup() {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      assert(instance().executorpools.size() == cppuddle::max_number_gpus);
      instance().executorpools.clear();
    }

    static std::tuple<Interface &, size_t> get_interface(const size_t gpu_id = 0) {
      std::lock_guard<cppuddle::mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      assert(gpu_id < instance().executorpools.size());
      return instance().executorpools[gpu_id].get_interface();
    }
    static void release_interface(size_t index, const size_t gpu_id = 0) {
      std::lock_guard<cppuddle::mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      assert(gpu_id < instance().executorpools.size());
      instance().executorpools[gpu_id].release_interface(index);
    }
    static bool interface_available(size_t load_limit, const size_t gpu_id = 0) {
      std::lock_guard<cppuddle::mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      assert(gpu_id < instance().executorpools.size());
      return instance().executorpools[gpu_id].interface_available(load_limit);
    }
    static size_t get_current_load(const size_t gpu_id = 0) {
      std::lock_guard<cppuddle::mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      assert(gpu_id < instance().executorpools.size());
      return instance().executorpools[gpu_id].get_current_load();
    }
    // TODO deprecated! Remove...
    /* static size_t get_next_device_id(const size_t gpu_id = 0) { */
    /*   std::lock_guard<cppuddle::mutex_t> guard(instance().gpu_mutexes[gpu_id]); */
    /*   assert(instance().executorpools.size() == cppuddle::max_number_gpus); */
    /*   return instance().executorpools[gpu_id].get_next_device_id(); */
    /* } */

    static void set_device_selector(std::function<void(size_t)> select_gpu_function) {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      instance().select_gpu_function = select_gpu_function;
    }

    static void select_device(size_t gpu_id) {
      instance().select_gpu_function(gpu_id);
    }

  private:
    executor_pool_implementation() = default;
    cppuddle::mutex_t pool_mut{};
    std::function<void(size_t)> select_gpu_function = [](size_t gpu_id) {
      // By default no multi gpu support
      assert(cppuddle::max_number_gpus == 1 || instance().executorpools.size() == 1);
      assert(gpu_id == 0);
    };

    std::deque<Pool> executorpools{};
    std::array<cppuddle::mutex_t, cppuddle::max_number_gpus> gpu_mutexes;

    static executor_pool_implementation& instance(void) {
      static executor_pool_implementation pool_instance{};
      return pool_instance;
    }

  public:
    ~executor_pool_implementation() = default;
    // Bunch of constructors we don't need
    executor_pool_implementation(executor_pool_implementation const &other) =
        delete;
    executor_pool_implementation &
    operator=(executor_pool_implementation const &other) = delete;
    executor_pool_implementation(executor_pool_implementation &&other) = delete;
    executor_pool_implementation &
    operator=(executor_pool_implementation &&other) = delete;
  };

public:
  ~executor_pool() = default;
  // Bunch of constructors we don't need
  executor_pool(executor_pool const &other) = delete;
  executor_pool &operator=(executor_pool const &other) = delete;
  executor_pool(executor_pool &&other) = delete;
  executor_pool &operator=(executor_pool &&other) = delete;
};

#if defined(CPPUDDLE_DEACTIVATE_EXECUTOR_RECYCLING)

// Warn about suboptimal performance without recycling
#pragma message                                                                \
"Warning: Building without executor recycling! Use only for performance testing! \
For better performance configure CPPuddle with CPPUDDLE_WITH_EXECUTOR_RECYCLING=ON!"

/// Slow version of the executor_interface that does not draw its
/// executors (Interface) from the pool but creates them instead.
/// Only meant for performance comparisons and only works with cuda/kokkos executors
template <typename Interface, typename Pool> class executor_interface {
public:

  template <typename Dummy = Interface>
  explicit executor_interface(size_t gpu_id,
      std::enable_if_t<std::is_same<hpx::cuda::experimental::cuda_executor, Dummy>::value, size_t> = 0)
      : gpu_id(gpu_id), interface(gpu_id) {}
  template <typename Dummy = Interface>
  explicit executor_interface(std::enable_if_t<!std::is_same<hpx::cuda::experimental::cuda_executor, Dummy>::value, size_t> = 0)
      : gpu_id(gpu_id), interface(hpx::kokkos::execution_space_mode::independent) {}

  executor_interface(const executor_interface &other) = delete;
  executor_interface &operator=(const executor_interface &other) = delete;
  executor_interface(executor_interface &&other) = delete;
  executor_interface &operator=(executor_interface &&other) = delete;
  ~executor_interface() {
  }

  template <typename F, typename... Ts>
  inline decltype(auto) post(F &&f, Ts &&... ts) {
    return interface.post(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  template <typename F, typename... Ts>
  inline decltype(auto) async_execute(F &&f, Ts &&... ts) {
    return interface.async_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  inline decltype(auto) get_future() {
    return interface.get_future();
  }

  // allow implict conversion
  operator Interface &() { // NOLINT
    return interface;
  }

private:
  size_t gpu_id;

public:
  Interface interface;
};
#else
/// Stream interface for RAII purposes
/// Draws executor from the executor pool and releases it upon
/// destruction
template <typename Interface, typename Pool> class executor_interface {
public:
  explicit executor_interface(size_t gpu_id)
      : t(executor_pool::get_interface<Interface, Pool>(gpu_id)),
        interface(std::get<0>(t)), interface_index(std::get<1>(t)), gpu_id(gpu_id) {}

  executor_interface(const executor_interface &other) = delete;
  executor_interface &operator=(const executor_interface &other) = delete;
  executor_interface(executor_interface &&other) = delete;
  executor_interface &operator=(executor_interface &&other) = delete;
  ~executor_interface() {
    executor_pool::release_interface<Interface, Pool>(interface_index, gpu_id);
  }

  template <typename F, typename... Ts>
  inline decltype(auto) post(F &&f, Ts &&... ts) {
    return interface.post(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  template <typename F, typename... Ts>
  inline decltype(auto) async_execute(F &&f, Ts &&... ts) {
    return interface.async_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  inline decltype(auto) get_future() {
    return interface.get_future();
  }

  // allow implict conversion
  operator Interface &() { // NOLINT
    return interface;
  }

private:
  std::tuple<Interface &, size_t> t;
  size_t interface_index;
  size_t gpu_id;

public:
  Interface &interface;
};
#endif

} // namespace detail
} // namespace executor_recycling
} // namespace cppuddle

#endif
