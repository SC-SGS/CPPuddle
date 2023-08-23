// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

#include <algorithm>
#include <cassert>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <functional>

//#include <cuda_runtime.h>
// #include <hpx/compute/cuda/target.hpp>
// #include <hpx/include/compute.hpp>

template <class Interface> class round_robin_pool {
private:
  std::deque<Interface> pool{};
  std::vector<size_t> ref_counters{};
  size_t current_interface{0};

public:
  template <typename... Ts>
  explicit round_robin_pool(size_t number_of_streams, Ts &&... executor_args) {
    ref_counters.reserve(number_of_streams);
    for (int i = 0; i < number_of_streams; i++) {
      pool.emplace_back(std::forward<Ts>(executor_args)...);
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
  explicit priority_pool(size_t number_of_streams, Ts &&... executor_args) {
    ref_counters.reserve(number_of_streams);
    priorities.reserve(number_of_streams);
    for (auto i = 0; i < number_of_streams; i++) {
      pool.emplace_back(std::forward<Ts>(executor_args)...);
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
                             Ts &&... executor_args)
      : streams_per_gpu{number_of_streams} {
    for (auto gpu_id = 0; gpu_id < number_of_gpus; gpu_id++) {
      pool.push_back(std::make_tuple(
          Pool(number_of_streams, gpu_id, std::forward<Ts>(executor_args)...),
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
                          Ts &&... executor_args)
      : streams_per_gpu(number_of_streams) {
    ref_counters.reserve(number_of_gpus);
    priorities.reserve(number_of_gpus);
    for (auto gpu_id = 0; gpu_id < number_of_gpus; gpu_id++) {
      priorities.emplace_back(gpu_id);
      ref_counters.emplace_back(0);
      gpu_interfaces.emplace_back(streams_per_gpu, gpu_id,
                                  std::forward<Ts>(executor_args)...);
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
  static void init(size_t number_of_streams, Ts &&... executor_args) {
    std::lock_guard<std::mutex> guard(mut);
    if (!access_instance) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      access_instance.reset(new stream_pool());
    }
    assert(access_instance); 
    stream_pool_implementation<Interface, Pool>::init(
        number_of_streams, std::forward<Ts>(executor_args)...);
  }
  // Dummy for interface compatbility with future 0.3.0 release
  // Works the same as the init method here
  template <class Interface, class Pool, typename... Ts>
  static void init_all_executor_pools(size_t number_of_streams, Ts &&... executor_args) {
    std::lock_guard<std::mutex> guard(mut);
    if (!access_instance) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      access_instance.reset(new stream_pool());
    }
    assert(access_instance); 
    stream_pool_implementation<Interface, Pool>::init(
        number_of_streams, std::forward<Ts>(executor_args)...);
  }
  // Dummy for interface compatbility with future 0.3.0 release
  // Works the same as the init method here
  template <class Interface, class Pool, typename... Ts>
  static void init_executor_pool(size_t device_id, size_t number_of_streams, Ts &&... executor_args) {
    if (device_id > 0)
      throw std::runtime_error("Got device_id > 0. MultiGPU not yet supported in cppuddle v0.2.1");
    std::lock_guard<std::mutex> guard(mut);
    if (!access_instance) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      access_instance.reset(new stream_pool());
    }
    assert(access_instance); 
    stream_pool_implementation<Interface, Pool>::init(
        number_of_streams, std::forward<Ts>(executor_args)...);
  }
  template <class Interface, class Pool> static void cleanup() {
    assert(access_instance); // should already be initialized
    stream_pool_implementation<Interface, Pool>::cleanup();
  }
  template <class Interface, class Pool>
  static std::tuple<Interface &, size_t> get_interface(size_t device_id = 0) {
    if (device_id > 0)
      throw std::runtime_error("Got device_id > 0. MultiGPU not yet supported in cppuddle v0.2.1");
    assert(access_instance); // should already be initialized
    return stream_pool_implementation<Interface, Pool>::get_interface();
  }
  template <class Interface, class Pool>
  static void release_interface(size_t index, size_t device_id = 0) {
    if (device_id > 0)
      throw std::runtime_error("Got device_id > 0. MultiGPU not yet supported in cppuddle v0.2.1");
    assert(access_instance); // should already be initialized
    stream_pool_implementation<Interface, Pool>::release_interface(index);
  }
  template <class Interface, class Pool>
  static bool interface_available(size_t load_limit, size_t device_id = 0) {
    if (device_id > 0)
      throw std::runtime_error("Got device_id > 0. MultiGPU not yet supported in cppuddle v0.2.1");
    assert(access_instance); // should already be initialized
    return stream_pool_implementation<Interface, Pool>::interface_available(
        load_limit);
  }
  template <class Interface, class Pool>
  static size_t get_current_load(size_t device_id = 0) {
    if (device_id > 0)
      throw std::runtime_error("Got device_id > 0. MultiGPU not yet supported in cppuddle v0.2.1");
    assert(access_instance); // should already be initialized
    return stream_pool_implementation<Interface, Pool>::get_current_load();
  }
  template <class Interface, class Pool>
  static size_t get_next_device_id(size_t num_devices = 1) {
    if (num_devices > 1)
      throw std::runtime_error("Got num_devices > 1. MultiGPU not yet supported in cppuddle v0.2.1");
    assert(access_instance); // should already be initialized
    return 0;
  }

  template <class Interface, class Pool>
  static size_t select_device(size_t device_id = 0) {
    if (device_id > 0)
      throw std::runtime_error("Got device_id > 0. MultiGPU not yet supported in cppuddle v0.2.1");
    return 0;
  }

  // Dummy for interface compatbility with future 0.3.0 release
  template <class Interface, class Pool>
  static void set_device_selector(std::function<void(size_t)> select_gpu_function) {
    
  }

private:
  static std::unique_ptr<stream_pool> access_instance;
  static std::mutex mut;
  stream_pool() = default;

private:
  template <class Interface, class Pool> class stream_pool_implementation {
  public:
    template <typename... Ts>
    static void init(size_t number_of_streams, Ts &&... executor_args) {
      // TODO(daissgr) What should happen if the instance already exists?
      // warning?
      if (!pool_instance && number_of_streams > 0) {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        pool_instance.reset(new stream_pool_implementation());
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        pool_instance->streampool.reset(
            new Pool{number_of_streams, std::forward<Ts>(executor_args)...});
      }
    }
    static void cleanup() {
      std::lock_guard<std::mutex> guard(pool_mut);
      if (pool_instance) {
        pool_instance->streampool.reset(nullptr);
        pool_instance.reset(nullptr);
      }
    }

    static std::tuple<Interface &, size_t> get_interface() noexcept {
      std::lock_guard<std::mutex> guard(pool_mut);
      assert(pool_instance); // should already be initialized
      return pool_instance->streampool->get_interface();
    }
    static void release_interface(size_t index) noexcept {
      std::lock_guard<std::mutex> guard(pool_mut);
      assert(pool_instance); // should already be initialized
      pool_instance->streampool->release_interface(index);
    }
    static bool interface_available(size_t load_limit) noexcept {
      std::lock_guard<std::mutex> guard(pool_mut);
      if (!pool_instance) {
        return false;
      }
      return pool_instance->streampool->interface_available(load_limit);
    }
    static size_t get_current_load() noexcept {
      std::lock_guard<std::mutex> guard(pool_mut);
      if (!pool_instance) {
        return 0;
      }
      assert(pool_instance); // should already be initialized
      return pool_instance->streampool->get_current_load();
    }
    static size_t get_next_device_id() noexcept {
      std::lock_guard<std::mutex> guard(pool_mut);
      if (!pool_instance) {
        return 0;
      }
      return pool_instance->streampool->get_next_device_id();
    }

  private:
    static std::unique_ptr<stream_pool_implementation> pool_instance;
    stream_pool_implementation() = default;
    inline static std::mutex pool_mut{};

    std::unique_ptr<Pool> streampool{nullptr};

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

template <class Interface, class Pool>
std::unique_ptr<stream_pool::stream_pool_implementation<Interface, Pool>>
    stream_pool::stream_pool_implementation<Interface, Pool>::pool_instance{};

template <class Interface, class Pool> class stream_interface {
public:
  explicit stream_interface(size_t device_id = 0)
      : t(stream_pool::get_interface<Interface, Pool>()),
        interface(std::get<0>(t)), interface_index(std::get<1>(t)) {
    if (device_id > 0)
      throw std::runtime_error("Got device_id > 0. MultiGPU not yet supported in cppuddle v0.2.1");
  }

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
