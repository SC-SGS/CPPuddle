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
  using interface_entry =
      std::tuple<Interface, size_t>; // interface, ref counter
  std::vector<interface_entry> pool{};
  size_t current_interface{0};

public:
  round_robin_pool(size_t gpu_id, size_t number_of_streams) {
    pool = std::vector<interface_entry>(number_of_streams,
                                        std::make_tuple(Interface(gpu_id), 0));
  }
  // return a tuple with the interface and its index (to release it later)
  interface_entry get_interface() {
    size_t last_interface = current_interface;
    current_interface = (current_interface + 1) % pool.size();
    std::get<1>(pool[last_interface])++;
    return std::make_tuple(std::get<0>(pool[last_interface]), last_interface);
  }
  void release_interface(size_t index) { std::get<1>(pool[index])--; }
  bool interface_available(size_t load_limit) {
    return std::get<1>(*(std::min_element(
               std::begin(pool), std::end(pool),
               [](const interface_entry &first,
                  const interface_entry &second) -> bool {
                 return std::get<1>(first) < std::get<1>(second);
               }))) < load_limit;
  }
  size_t get_current_load() {
    return std::get<1>(*(std::min_element(
        std::begin(pool), std::end(pool),
        [](const interface_entry &first, const interface_entry &second)
            -> bool { return std::get<1>(first) < std::get<1>(second); })));
  }
};

template <class Interface> class priority_pool {
private:
  using interface_entry =
      std::tuple<Interface, size_t>; // Interface, ID of ref counter field
  std::vector<interface_entry> pool{};
  std::vector<size_t> ref_counters{}; // Ref counters
public:
  priority_pool(size_t gpu_id, size_t number_of_streams) {
    for (auto i = 0; i < number_of_streams; i++) {
      pool.push_back(std::make_tuple(Interface(gpu_id), i));
      ref_counters.push_back(0);
    }
  }
  // return a tuple with the interface and its index (to release it later)
  std::tuple<Interface, size_t> get_interface() {
    auto interface = pool[0];
    ref_counters[std::get<1>(interface)]++;
    std::make_heap(std::begin(pool), std::end(pool),
                   [this](const interface_entry &first,
                          const interface_entry &second) -> bool {
                     return ref_counters[std::get<1>(first)] >
                            ref_counters[std::get<1>(second)];
                   });
    return interface;
  }
  void release_interface(size_t index) {
    ref_counters[index]--;
    std::make_heap(std::begin(pool), std::end(pool),
                   [this](const interface_entry &first,
                          const interface_entry &second) -> bool {
                     return ref_counters[std::get<1>(first)] >
                            ref_counters[std::get<1>(second)];
                   });
  }
  bool interface_available(size_t load_limit) {
    return ref_counters[std::get<1>(pool[0])] < load_limit;
  }
  size_t get_current_load() { return ref_counters[std::get<1>(pool[0])]; }
};

template <class Interface, class Pool> class multi_gpu_round_robin_pool {
private:
  using interface_entry =
      std::tuple<Interface, size_t>;          // interface, ref counter
  using gpu_entry = std::tuple<Pool, size_t>; // interface, ref counter
  std::vector<gpu_entry> pool{};
  size_t current_interface{0};
  size_t streams_per_gpu{0};

public:
  multi_gpu_round_robin_pool(size_t number_of_gpus, size_t number_of_streams)
      : streams_per_gpu{number_of_streams} {
    for (size_t gpu_id = 0; gpu_id < number_of_gpus; gpu_id++) {
      pool.push_back(std::make_tuple(Pool(gpu_id, number_of_streams), 0));
    }
  }

  // return a tuple with the interface and its index (to release it later)
  interface_entry get_interface() {
    size_t last_interface = current_interface;
    current_interface = (current_interface + 1) % pool.size();
    std::get<1>(pool[last_interface])++;
    size_t gpu_offset = last_interface * streams_per_gpu;
    auto stream_entry = std::get<0>(pool[last_interface]).get_interface();
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
  using interface_entry = size_t;
  std::vector<interface_entry> pool{};
  std::vector<size_t> ref_counters{};
  std::vector<Pool> gpu_interfaces{};
  size_t streams_per_gpu{0};

public:
  priority_pool_multi_gpu(size_t number_of_gpus, size_t number_of_streams)
      : streams_per_gpu(number_of_streams) {
    for (auto i = 0; i < number_of_gpus; i++) {
      pool.push_back(i);
      ref_counters.push_back(0);
      gpu_interfaces.push_back(Pool(i, streams_per_gpu));
    }
  }
  // return a tuple with the interface and its index (to release it later)
  std::tuple<Interface, size_t> get_interface() {
    auto gpu = pool[0];
    ref_counters[gpu]++;
    std::make_heap(std::begin(pool), std::end(pool),
                   [this](const interface_entry &first,
                          const interface_entry &second) -> bool {
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
    std::make_heap(std::begin(pool), std::end(pool),
                   [this](const interface_entry &first,
                          const interface_entry &second) -> bool {
                     return ref_counters[first] > ref_counters[second];
                   });
    gpu_interfaces[gpu_index].release_interface(stream_index);
  }
  bool interface_available(size_t load_limit) {
    return gpu_interfaces[pool[0]].interface_available(load_limit);
  }
  size_t get_current_load() {
    return gpu_interfaces[pool[0]].get_current_load();
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
  static std::tuple<Interface, size_t> get_interface() {
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
      if (!pool_instance) {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        pool_instance.reset(new stream_pool_implementation());
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        pool_instance->streampool.reset(new Pool{gpu_id, number_of_streams});
      }
    }
    static std::tuple<Interface, size_t> get_interface() {
      assert(pool_instance); // should already be initialized
      return pool_instance->streampool->get_interface();
    }
    static void release_interface(size_t index) noexcept {
      assert(pool_instance); // should already be initialized
      pool_instance->streampool->release_interface(index);
    }
    static bool interface_available(size_t load_limit) noexcept {
      assert(pool_instance); // should already be initialized
      return pool_instance->streampool->interface_available(load_limit);
    }
    static size_t get_current_load() noexcept {
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
    stream_pool_implementation
    operator=(stream_pool_implementation const &other) = delete;
    stream_pool_implementation(stream_pool_implementation &&other) = delete;
    stream_pool_implementation
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
  explicit stream_interface(std::size_t device_id) {
    auto tmp_interface_tuple = stream_pool::get_interface<Interface, Pool>();
    interface = std::get<0>(tmp_interface_tuple);
    interface_index = std::get<1>(tmp_interface_tuple);
  }

  stream_interface(const stream_interface &other) = delete;
  stream_interface &operator=(const stream_interface &other) = delete;
  stream_interface(stream_interface &&other) = delete;
  stream_interface &operator=(stream_interface &&other) = delete;
  ~stream_interface() {
    stream_pool::release_interface<Interface, Pool>(interface_index);
  }

  template <typename... Args> cudaError_t pass_through(Args &&... args) {
    return interface.pass_through(std::forward<Args>(args)...);
  }
  template <typename... Args> void execute(Args &&... args) {
    return interface.execute(std::forward<Args>(args)...);
  }

  template <typename... Args> void copy_async(Args &&... args) {
    interface.copy_async(std::forward<Args>(args)...);
  }
  template <typename... Args> void memset_async(Args &&... args) {
    interface.memset_async(std::forward<Args>(args)...);
  }
  template <typename future_type> future_type get_future() {
    return interface.get_future();
  }

private:
  cuda_helper interface{};
  size_t interface_index{0};
};

using hpx_stream_interface_pq =
    stream_interface<cuda_helper, priority_pool<cuda_helper>>;
using hpx_stream_interface_mgpq = stream_interface<
    cuda_helper,
    priority_pool_multi_gpu<cuda_helper, priority_pool<cuda_helper>>>;
using hpx_stream_interface_rr =
    stream_interface<cuda_helper, round_robin_pool<cuda_helper>>;
using hpx_stream_interface_mgrr = stream_interface<
    cuda_helper,
    multi_gpu_round_robin_pool<cuda_helper, round_robin_pool<cuda_helper>>>;
#endif