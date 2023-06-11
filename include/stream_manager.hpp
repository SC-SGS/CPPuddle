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
  size_t get_next_device_id() {
    return 0; // single gpu pool
  }
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
  size_t get_next_device_id() {
    return 0; // single gpu pool
  }
};

template <class Interface, class Pool> class multi_gpu_round_robin_pool {
private:
  using gpu_entry = std::tuple<Pool, size_t>; // interface, ref counter
  std::deque<gpu_entry> pool{};
  size_t current_interface{0};
  size_t streams_per_gpu{0};

public:
  template <typename... Ts>
  multi_gpu_round_robin_pool(size_t number_of_streams, int number_of_gpus,
                             Ts... executor_args)
      : streams_per_gpu{number_of_streams} {
    for (auto gpu_id = 0; gpu_id < number_of_gpus; gpu_id++) {
      pool.push_back(std::make_tuple(
          Pool(number_of_streams, gpu_id, executor_args...),
          0));
    }
  }

  // return a tuple with the interface and its index (to release it later)
  std::tuple<Interface &, size_t> get_interface() {
    size_t last_interface = current_interface;
    current_interface = (current_interface + 1) % pool.size();
    std::get<1>(pool[last_interface])++;
    size_t gpu_offset = last_interface * streams_per_gpu;
    std::tuple<Interface &, size_t> stream_entry =
        std::get<0>(pool[last_interface]).get_interface();
    std::get<1>(stream_entry) += gpu_offset;
    return stream_entry;
  }
  void release_interface(size_t index) {
    size_t gpu_index = index / streams_per_gpu;
    size_t stream_index = index % streams_per_gpu;
    std::get<1>(pool[gpu_index])--;
    std::get<0>(pool[gpu_index]).release_interface(stream_index);
  }
  bool interface_available(size_t load_limit) {
    auto &current_min_gpu = std::get<0>(*(std::min_element(
        std::begin(pool), std::end(pool),
        [](const gpu_entry &first, const gpu_entry &second) -> bool {
          return std::get<1>(first) < std::get<1>(second);
        })));
    return current_min_gpu.interface_available(load_limit);
  }
  size_t get_current_load() {
    auto &current_min_gpu = std::get<0>(*(std::min_element(
        std::begin(pool), std::end(pool),
        [](const gpu_entry &first, const gpu_entry &second) -> bool {
          return std::get<1>(first) < std::get<1>(second);
        })));
    return current_min_gpu.get_current_load();
  }
  size_t get_next_device_id() { return current_interface; }
};

template <class Interface, class Pool> class priority_pool_multi_gpu {
private:
  std::vector<size_t> priorities{};
  std::vector<size_t> ref_counters{};
  std::deque<Pool> gpu_interfaces{};
  size_t streams_per_gpu{0};

public:
  template <typename... Ts>
  priority_pool_multi_gpu(size_t number_of_streams, int number_of_gpus,
                          Ts... executor_args)
      : streams_per_gpu(number_of_streams) {
    ref_counters.reserve(number_of_gpus);
    priorities.reserve(number_of_gpus);
    for (auto gpu_id = 0; gpu_id < number_of_gpus; gpu_id++) {
      priorities.emplace_back(gpu_id);
      ref_counters.emplace_back(0);
      gpu_interfaces.emplace_back(streams_per_gpu, gpu_id,
                                  executor_args...);
    }
  }
  // return a tuple with the interface and its index (to release it later)
  std::tuple<Interface &, size_t> get_interface() {
    auto gpu = priorities[0];
    ref_counters[gpu]++;
    std::make_heap(std::begin(priorities), std::end(priorities),
                   [this](const size_t &first, const size_t &second) -> bool {
                     return ref_counters[first] > ref_counters[second];
                   });
    size_t gpu_offset = gpu * streams_per_gpu;
    auto stream_entry = gpu_interfaces[gpu].get_interface();
    std::get<1>(stream_entry) += gpu_offset;
    return stream_entry;
  }
  void release_interface(size_t index) {
    size_t gpu_index = index / streams_per_gpu;
    size_t stream_index = index % streams_per_gpu;
    ref_counters[gpu_index]--;
    std::make_heap(std::begin(priorities), std::end(priorities),
                   [this](const size_t &first, const size_t &second) -> bool {
                     return ref_counters[first] > ref_counters[second];
                   });
    gpu_interfaces[gpu_index].release_interface(stream_index);
  }
  bool interface_available(size_t load_limit) {
    return gpu_interfaces[priorities[0]].interface_available(load_limit);
  }
  size_t get_current_load() {
    return gpu_interfaces[priorities[0]].get_current_load();
  }
  size_t get_next_device_id() { return priorities[0]; }
};

/// Access/Concurrency Control for stream pool implementation
class stream_pool {
public:
  template <class Interface, class Pool, typename... Ts>
  static void init(size_t number_of_streams, Ts ... executor_args) {
    stream_pool_implementation<Interface, Pool>::init(number_of_streams,
                                                      executor_args...);
}
  template <class Interface, class Pool> static void cleanup() {
    stream_pool_implementation<Interface, Pool>::cleanup();
  }
  template <class Interface, class Pool>
  static std::tuple<Interface &, size_t> get_interface() {
    return stream_pool_implementation<Interface, Pool>::get_interface(get_device_id());
  }
  template <class Interface, class Pool>
  static void release_interface(size_t index) noexcept {
    stream_pool_implementation<Interface, Pool>::release_interface(index,
        get_device_id());
  }
  template <class Interface, class Pool>
  static bool interface_available(size_t load_limit) noexcept {
    return stream_pool_implementation<Interface, Pool>::interface_available(
        load_limit, get_device_id());
  }
  template <class Interface, class Pool>
  static size_t get_current_load() noexcept {
    return stream_pool_implementation<Interface, Pool>::get_current_load(
        get_device_id());
  }
  // TODO deprecated! Remove...
  template <class Interface, class Pool>
  static size_t get_next_device_id() noexcept {
    return stream_pool_implementation<Interface, Pool>::get_next_device_id(get_device_id());
  }

  template <class Interface, class Pool>
  static size_t set_device_selector(std::function<void(size_t)> select_gpu_function) {
    return stream_pool_implementation<Interface, Pool>::set_device_selector(select_gpu_function);
  }

private:
  stream_pool() = default;

private:
  template <class Interface, class Pool> class stream_pool_implementation {
  public:
    template <typename... Ts>
    static void init(size_t number_of_streams, Ts ... executor_args) {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      if (number_of_streams > 0) {
        for (size_t gpu_id = 0; gpu_id < max_number_gpus; gpu_id++) {
          instance().select_gpu_function(gpu_id);
          instance().streampools.emplace_back(number_of_streams,
                                              executor_args...);
        }
      }
    }

    // TODO add/rename into finalize?
    static void cleanup() {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      instance().streampools.clear();
    }

    static std::tuple<Interface &, size_t> get_interface(const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      return instance().streampools[gpu_id].get_interface();
    }
    static void release_interface(size_t index, const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      instance().streampools[gpu_id].release_interface(index);
    }
    static bool interface_available(size_t load_limit, const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      return instance().streampools[gpu_id].interface_available(load_limit);
    }
    static size_t get_current_load(const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      return instance().streampools[gpu_id].get_current_load();
    }
    // TODO deprecated! Remove...
    static size_t get_next_device_id(const size_t gpu_id = 0) {
      std::lock_guard<mutex_t> guard(instance().gpu_mutexes[gpu_id]);
      return instance().streampools[gpu_id].get_next_device_id();
    }

    static size_t set_device_selector(std::function<void(size_t)> select_gpu_function) {
      auto guard = make_scoped_lock_from_array(instance().gpu_mutexes);
      return instance().select_gpu_function = select_gpu_function;
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
  stream_interface()
      : t(stream_pool::get_interface<Interface, Pool>()),
        interface(std::get<0>(t)), interface_index(std::get<1>(t)) {}

  stream_interface(const stream_interface &other) = delete;
  stream_interface &operator=(const stream_interface &other) = delete;
  stream_interface(stream_interface &&other) = delete;
  stream_interface &operator=(stream_interface &&other) = delete;
  ~stream_interface() {
    stream_pool::release_interface<Interface, Pool>(interface_index);
  }

  template <typename F, typename... Ts>
  inline decltype(auto) post(F &&f, Ts &&... ts) {
    return interface.post(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  template <typename F, typename... Ts>
  inline decltype(auto) async_execute(F &&f, Ts &&... ts) {
    return interface.async_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  inline size_t get_gpu_id() noexcept { return interface.get_gpu_id(); }

  // allow implict conversion
  operator Interface &() { // NOLINT
    return interface;
  }

private:
  std::tuple<Interface &, size_t> t;
  size_t interface_index;

public:
  Interface &interface;
};

#endif
