// Copyright (c) 2020-2023 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

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

#include "../include/detail/config.hpp"

/// Turns a std::array_mutex into an scoped lock
template<typename mutex_array_t>
auto make_scoped_lock_from_array(mutex_array_t& mutexes)
{
    return std::apply([](auto&... mutexes) { return std::scoped_lock{mutexes...}; }, 
                      mutexes);
}

template <class Interface> class round_robin_pool {
private:
  std::deque<Interface> pool{};
  std::vector<size_t> ref_counters{};
  size_t current_interface{0};

public:
  template <typename... Ts>
  round_robin_pool(size_t number_of_streams, Ts... executor_args) {
    ref_counters.reserve(number_of_streams);
    for (int i = 0; i < number_of_streams; i++) {
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

template <class Interface> class priority_pool {
private:
  std::deque<Interface> pool{};
  std::vector<size_t> ref_counters{}; // Ref counters
  std::vector<size_t> priorities{};   // Ref counters
public:
  template <typename... Ts>
  priority_pool(size_t number_of_streams, Ts... executor_args) {
    ref_counters.reserve(number_of_streams);
    priorities.reserve(number_of_streams);
    for (auto i = 0; i < number_of_streams; i++) {
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

/// Access/Concurrency Control for stream pool implementation
class stream_pool {
public:
  template <class Interface, class Pool, typename... Ts>
  static void init(size_t number_of_streams, Ts ... executor_args) {
    stream_pool_implementation<Interface, Pool>::init(number_of_streams,
                                                      executor_args...);
  }
  template <class Interface, class Pool, typename... Ts>
  static void init_all_executor_pools(size_t number_of_streams, Ts ... executor_args) {
    stream_pool_implementation<Interface, Pool>::init_all_executor_pools(number_of_streams,
                                                      executor_args...);
  }
  template <class Interface, class Pool, typename... Ts>
  static void init_executor_pool(size_t pool_id, size_t number_of_streams, Ts ... executor_args) {
    stream_pool_implementation<Interface, Pool>::init_executor_pool(pool_id, number_of_streams,
                                                      executor_args...);
  }
  template <class Interface, class Pool> static void cleanup() {
    stream_pool_implementation<Interface, Pool>::cleanup();
  }
  template <class Interface, class Pool>
  static std::tuple<Interface &, size_t> get_interface(const size_t gpu_id) {
    return stream_pool_implementation<Interface, Pool>::get_interface(gpu_id);
  }
  template <class Interface, class Pool>
  static void release_interface(size_t index, const size_t gpu_id) noexcept {
    stream_pool_implementation<Interface, Pool>::release_interface(index,
        gpu_id);
  }
  template <class Interface, class Pool>
  static bool interface_available(size_t load_limit, const size_t gpu_id) noexcept {
    return stream_pool_implementation<Interface, Pool>::interface_available(
        load_limit, gpu_id);
  }
  template <class Interface, class Pool>
  static size_t get_current_load(const size_t gpu_id = 0) noexcept {
    return stream_pool_implementation<Interface, Pool>::get_current_load(
        gpu_id);
  }
  template <class Interface, class Pool>
  static size_t get_next_device_id(const size_t number_gpus) noexcept {
    // TODO add round robin and min strategy
    return get_device_id(number_gpus);
  }

  template <class Interface, class Pool>
  static void set_device_selector(std::function<void(size_t)> select_gpu_function) {
    stream_pool_implementation<Interface, Pool>::set_device_selector(select_gpu_function);
  }

  template <class Interface, class Pool>
  static void select_device(size_t gpu_id) {
    stream_pool_implementation<Interface, Pool>::select_device(gpu_id);
  }

private:
  stream_pool() = default;

private:
  template <class Interface, class Pool> class stream_pool_implementation {
  public:
    /// Deprecated! Use init_on_all_gpu or init_on_gpu
    template <typename... Ts>
    static void init(size_t number_of_streams, Ts ... executor_args) {
      /* static_assert(sizeof...(Ts) == sizeof...(Ts) && max_number_gpus == 1, */
      /*               "deprecated stream_pool::init does not support multigpu"); */
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      instance().streampools.emplace_back(number_of_streams, executor_args...);
    }

    /// Multi-GPU init where executors / interfaces on all GPUs are initialized with the same arguments
    template <typename... Ts>
    static void init_all_executor_pools(size_t number_of_streams, Ts ... executor_args) {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      if (number_of_streams > 0) {
        for (size_t gpu_id = 0; gpu_id < max_number_gpus; gpu_id++) {
          instance().select_gpu_function(gpu_id);
          instance().streampools.emplace_back(number_of_streams,
                                              executor_args...);
        }
      }
    }

    /// Per-GPU init allowing for different init parameters depending on the GPU 
    /// (useful for executor that expect an GPU-id during construction)
    template <typename... Ts>
    static void init_executor_pool(size_t gpu_id, size_t number_of_streams, Ts ... executor_args) {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      if (number_of_streams > 0) {
        instance().select_gpu_function(gpu_id);
        instance().streampools.emplace_back(number_of_streams, 
                                            executor_args...);
      }
    }

    // TODO add/rename into finalize?
    static void cleanup() {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      assert(instance().streampools.size() == max_number_gpus);
      instance().streampools.clear();
    }

    static std::tuple<Interface &, size_t> get_interface(const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      assert(instance().streampools.size() == max_number_gpus);
      return instance().streampools[gpu_id].get_interface();
    }
    static void release_interface(size_t index, const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      assert(instance().streampools.size() == max_number_gpus);
      instance().streampools[gpu_id].release_interface(index);
    }
    static bool interface_available(size_t load_limit, const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      assert(instance().streampools.size() == max_number_gpus);
      return instance().streampools[gpu_id].interface_available(load_limit);
    }
    static size_t get_current_load(const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      assert(instance().streampools.size() == max_number_gpus);
      return instance().streampools[gpu_id].get_current_load();
    }
    // TODO deprecated! Remove...
    /* static size_t get_next_device_id(const size_t gpu_id = 0) { */
    /*   std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]); */
    /*   assert(instance().streampools.size() == max_number_gpus); */
    /*   return instance().streampools[gpu_id].get_next_device_id(); */
    /* } */

    static void set_device_selector(std::function<void(size_t)> select_gpu_function) {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      assert(instance().streampools.size() == max_number_gpus);
      instance().select_gpu_function = select_gpu_function;
    }

    static void select_device(size_t gpu_id) {
      instance().select_gpu_function(gpu_id);
    }

  private:
    stream_pool_implementation() = default;
    mutex_t pool_mut{};
    std::function<void(size_t)> select_gpu_function = [](size_t gpu_id) {
      // By default no multi gpu support
      assert(max_number_gpus == 1);
      assert(gpu_id == 0);
    };

    std::deque<Pool> streampools{};
    std::array<mutex_t, max_number_gpus> gpu_mutexes;

    static stream_pool_implementation& instance(void) {
      static stream_pool_implementation pool_instance{};
      return pool_instance;
    }

  public:
    ~stream_pool_implementation() = default;
    // Bunch of constructors we don't need
    stream_pool_implementation(stream_pool_implementation const &other) =
        delete;
    stream_pool_implementation &
    operator=(stream_pool_implementation const &other) = delete;
    stream_pool_implementation(stream_pool_implementation &&other) = delete;
    stream_pool_implementation &
    operator=(stream_pool_implementation &&other) = delete;
  };

public:
  ~stream_pool() = default;
  // Bunch of constructors we don't need
  stream_pool(stream_pool const &other) = delete;
  stream_pool &operator=(stream_pool const &other) = delete;
  stream_pool(stream_pool &&other) = delete;
  stream_pool &operator=(stream_pool &&other) = delete;
};

template <class Interface, class Pool> class stream_interface {
public:
  explicit stream_interface(size_t gpu_id)
      : t(stream_pool::get_interface<Interface, Pool>(gpu_id)),
        interface(std::get<0>(t)), interface_index(std::get<1>(t)), gpu_id(gpu_id) {}

  stream_interface(const stream_interface &other) = delete;
  stream_interface &operator=(const stream_interface &other) = delete;
  stream_interface(stream_interface &&other) = delete;
  stream_interface &operator=(stream_interface &&other) = delete;
  ~stream_interface() {
    stream_pool::release_interface<Interface, Pool>(interface_index, gpu_id);
  }

  template <typename F, typename... Ts>
  inline decltype(auto) post(F &&f, Ts &&... ts) {
    return interface.post(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  template <typename F, typename... Ts>
  inline decltype(auto) async_execute(F &&f, Ts &&... ts) {
    return interface.async_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
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
