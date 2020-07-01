#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <type_traits>

#include <cuda_runtime.h>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/include/compute.hpp>

#include "cuda_helper.hpp"

template <class Interface> class round_robin_pool {
private:
  std::vector<Interface> pool{};
  std::vector<size_t> ref_counters{};
  size_t current_interface{0};

public:
  round_robin_pool(size_t gpu_id, size_t number_of_streams) {
    pool.reserve(number_of_streams);
    ref_counters.reserve(number_of_streams);
    for (int i = 0; i < number_of_streams; i++) {
      pool.emplace_back(gpu_id, false);
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
};

template <class Interface> class priority_pool {
private:
  std::vector<Interface> pool{};
  std::vector<size_t> ref_counters{}; // Ref counters
  std::vector<size_t> priorities{};   // Ref counters
public:
  priority_pool(size_t gpu_id, size_t number_of_streams) {
    pool.reserve(number_of_streams);
    for (auto i = 0; i < number_of_streams; i++) {
      pool.emplace_back(gpu_id, false);
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
};

template <class Interface, class Pool> class multi_gpu_round_robin_pool {
private:
  using gpu_entry = std::tuple<Pool, size_t>; // interface, ref counter
  std::vector<gpu_entry> pool{};
  size_t current_interface{0};
  size_t streams_per_gpu{0};

public:
  multi_gpu_round_robin_pool(size_t number_of_gpus, size_t number_of_streams)
      : streams_per_gpu{number_of_streams} {
    for (size_t gpu_id = 0; gpu_id < number_of_gpus; gpu_id++) {
      // TODO(daissgr) why not emplace back?
      pool.push_back(std::make_tuple(Pool(gpu_id, number_of_streams), 0));
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
};

template <class Interface, class Pool> class priority_pool_multi_gpu {
private:
  std::vector<size_t> priorities{};
  std::vector<size_t> ref_counters{};
  std::vector<Pool> gpu_interfaces{};
  size_t streams_per_gpu{0};

public:
  priority_pool_multi_gpu(size_t number_of_gpus, size_t number_of_streams)
      : streams_per_gpu(number_of_streams) {
    for (auto i = 0; i < number_of_gpus; i++) {
      priorities.emplace_back(i);
      ref_counters.emplace_back(0);
      gpu_interfaces.emplace_back(i, streams_per_gpu);
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
};

/// Access/Concurrency Control for stream pool implementation
class stream_pool {
public:
  template <class Interface, class Pool>
  static void init(size_t gpu_id, size_t number_of_streams) {
    std::lock_guard<std::mutex> guard(mut);
    if (!access_instance) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      access_instance.reset(new stream_pool());
    }
    stream_pool_implementation<Interface, Pool>::init(gpu_id,
                                                      number_of_streams);
  }
  template <class Interface, class Pool>
  static std::tuple<Interface &, size_t> get_interface() {
    std::lock_guard<std::mutex> guard(mut);
    assert(access_instance); // should already be initialized
    return stream_pool_implementation<Interface, Pool>::get_interface();
  }
  template <class Interface, class Pool>
  static void release_interface(size_t index) noexcept {
    std::lock_guard<std::mutex> guard(mut);
    assert(access_instance); // should already be initialized
    stream_pool_implementation<Interface, Pool>::release_interface(index);
  }
  template <class Interface, class Pool>
  static bool interface_available(size_t load_limit) noexcept {
    std::lock_guard<std::mutex> guard(mut);
    assert(access_instance); // should already be initialized
    return stream_pool_implementation<Interface, Pool>::interface_available(
        load_limit);
  }
  template <class Interface, class Pool>
  static size_t get_current_load() noexcept {
    std::lock_guard<std::mutex> guard(mut);
    assert(access_instance); // should already be initialized
    return stream_pool_implementation<Interface, Pool>::get_current_load();
  }

private:
  static std::unique_ptr<stream_pool> access_instance;
  static std::mutex mut;
  stream_pool() = default;

private:
  template <class Interface, class Pool> class stream_pool_implementation {
  public:
    static void init(size_t gpu_id, size_t number_of_streams) {
      if (!pool_instance && number_of_streams > 0) {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        pool_instance.reset(new stream_pool_implementation());
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        pool_instance->streampool.reset(new Pool{gpu_id, number_of_streams});
      }
    }
    static std::tuple<Interface &, size_t> get_interface() noexcept {
      assert(pool_instance); // should already be initialized
      return pool_instance->streampool->get_interface();
    }
    static void release_interface(size_t index) noexcept {
      assert(pool_instance); // should already be initialized
      pool_instance->streampool->release_interface(index);
    }
    static bool interface_available(size_t load_limit) noexcept {
      if (!pool_instance)
        return false;
      assert(pool_instance); // should already be initialized
      return pool_instance->streampool->interface_available(load_limit);
    }
    static size_t get_current_load() noexcept {
      if (!pool_instance)
        return 0;
      assert(pool_instance); // should already be initialized
      return pool_instance->streampool->get_current_load();
    }

  private:
    static std::unique_ptr<stream_pool_implementation> pool_instance;
    stream_pool_implementation() = default;

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
  explicit stream_interface()
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

  inline const size_t get_gpu_id() noexcept { return interface.get_gpu_id(); }

private:
  std::tuple<Interface &, size_t> t;
  Interface &interface;
  size_t interface_index;
};

// using hpx_stream_interface_pq =
//     stream_interface<cuda_helper, priority_pool<cuda_helper>>;
// using hpx_stream_interface_mgpq = stream_interface<
//     cuda_helper,
//     priority_pool_multi_gpu<cuda_helper, priority_pool<cuda_helper>>>;
// using hpx_stream_interface_rr =
//     stream_interface<cuda_helper, round_robin_pool<cuda_helper>>;
// using hpx_stream_interface_mgrr = stream_interface<
//     cuda_helper,
//     multi_gpu_round_robin_pool<cuda_helper, round_robin_pool<cuda_helper>>>;
#endif