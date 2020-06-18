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
  using interface_entry = std::tuple<Interface, size_t>;
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
  bool interface_available() { return true; }
};

template <class Interface> class priority_pool {
private:
  using interface_entry = std::tuple<Interface, size_t, size_t>;
  std::vector<interface_entry> pool{};
  std::vector<std::reference_wrapper<size_t>> ref_counters{};
  size_t current_interface{0};

public:
  priority_pool(size_t gpu_id, size_t number_of_streams) {
    for (auto i = 0; i < number_of_streams; i++) {
      pool.push_back(std::make_tuple(Interface(gpu_id), 0, i));
      ref_counters.push_back(std::get<2>(pool[i]));
    }
  }
  // return a tuple with the interface and its index (to release it later)
  interface_entry get_interface() {
    auto interface = pool[0];
    std::get<1>(interface)++;
    std::make_heap(std::begin(pool), std::end(pool),
                   [](const interface_entry &first,
                      const interface_entry &second) -> bool {
                     return std::get<1>(first) > std::get<1>(second);
                   });
    return std::make_tuple(std::get<0>(interface), std::get<2>(interface));
  }
  void release_interface(size_t index) {
    ref_counters[index]--;
    std::make_heap(std::begin(pool), std::end(pool),
                   [](const interface_entry &first,
                      const interface_entry &second) -> bool {
                     return std::get<1>(first) > std::get<1>(second);
                   });
  }
  bool interface_available() { return true; }
};

// balancing_round_robin_pool
// priority_pool
// balancing_priority_pool

/// Access/Concurrency Control for stream pool implementation
class stream_pool {
public:
  template <class Interface, class Pool>
  static void init(size_t gpu_id, size_t number_of_streams) {
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
  static void interface_available() noexcept {
    std::lock_guard<std::mutex> guard(mut);
    assert(access_instance); // should already be initialized
    stream_pool_implementation<Interface, Pool>::interface_available();
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
    static void interface_available(size_t index) noexcept {
      assert(pool_instance); // should already be initialized
      pool_instance->streampool->release_interface(index);
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

  // get the future to synchronize this cuBLAS stream with
  // future_type get_future() {
  //   return target_.get_future();
  // }

private:
  cuda_helper interface{};
  size_t interface_index{0};
};

using hpx_stream_interface =
    stream_interface<cuda_helper, round_robin_pool<cuda_helper>>;
#endif