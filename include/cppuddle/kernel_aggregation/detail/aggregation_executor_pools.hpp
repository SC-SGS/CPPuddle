// Copyright (c) 2022-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cppuddle/kernel_aggregation/detail/aggregation_executors_and_allocators.hpp"

#ifndef AGGREGATION_EXECUTOR_POOL_HPP
#define AGGREGATION_EXECUTOR_POOL_HPP

namespace cppuddle {
namespace kernel_aggregation {
namespace detail {

template <const char *kernelname, class Interface, class Pool>
class aggregation_pool {
public:
  /// interface
  template <typename... Ts>
  static void init(size_t number_of_executors, size_t slices_per_executor,
                   aggregated_executor_modes mode, size_t num_devices = 1) {
    if (is_initialized) {
      throw std::runtime_error(
          std::string("Trying to initialize cppuddle aggregation pool twice") +
          " Agg pool name: " + std::string(kernelname));
    }
    if (num_devices > cppuddle::max_number_gpus) {
      throw std::runtime_error(
          std::string(
              "Trying to initialize aggregation with more devices than the "
              "maximum number of GPUs given at compiletime") +
          " Agg pool name: " + std::string(kernelname));
    }
    number_devices = num_devices;
    for (size_t gpu_id = 0; gpu_id < number_devices; gpu_id++) {

      std::lock_guard<aggregation_mutex_t> guard(instance()[gpu_id].pool_mutex);
      assert(instance()[gpu_id].aggregation_executor_pool.empty());
      for (int i = 0; i < number_of_executors; i++) {
        instance()[gpu_id].aggregation_executor_pool.emplace_back(slices_per_executor,
                                                        mode, gpu_id);
      }
      instance()[gpu_id].slices_per_executor = slices_per_executor;
      instance()[gpu_id].mode = mode;
    }
    is_initialized = true;
  }

  /// Will always return a valid executor slice
  static decltype(auto) request_executor_slice(void) {
    if (!is_initialized) {
      throw std::runtime_error(
          std::string("Trying to use cppuddle aggregation pool without first calling init") +
          " Agg poolname: " + std::string(kernelname));
    }
    const size_t gpu_id = cppuddle::get_device_id(number_devices);
    /* const size_t gpu_id = 1; */
    std::lock_guard<aggregation_mutex_t> guard(instance()[gpu_id].pool_mutex);
    assert(!instance()[gpu_id].aggregation_executor_pool.empty());
    std::optional<hpx::lcos::future<
        typename aggregated_executor<Interface>::executor_slice>>
        ret;
    size_t local_id = (instance()[gpu_id].current_interface) %
                      instance()[gpu_id].aggregation_executor_pool.size();
    ret = instance()[gpu_id].aggregation_executor_pool[local_id].request_executor_slice();
    // Expected case: current aggregation executor is free
    if (ret.has_value()) {
      return ret;
    }
    // current interface is bad -> find free one
    size_t abort_counter = 0;
    const size_t abort_number = instance()[gpu_id].aggregation_executor_pool.size() + 1;
    do {
      local_id = (++(instance()[gpu_id].current_interface)) % // increment interface
                 instance()[gpu_id].aggregation_executor_pool.size();
      ret =
          instance()[gpu_id].aggregation_executor_pool[local_id].request_executor_slice();
      if (ret.has_value()) {
        return ret;
      }
      abort_counter++;
    } while (abort_counter <= abort_number);
    // Everything's busy -> create new aggregation executor (growing pool) OR
    // return empty optional
    if (instance()[gpu_id].growing_pool) {
      instance()[gpu_id].aggregation_executor_pool.emplace_back(
          instance()[gpu_id].slices_per_executor, instance()[gpu_id].mode, gpu_id);
      instance()[gpu_id].current_interface =
          instance()[gpu_id].aggregation_executor_pool.size() - 1;
      assert(instance()[gpu_id].aggregation_executor_pool.size() < 20480);
      ret = instance()[gpu_id]
                .aggregation_executor_pool[instance()[gpu_id].current_interface]
                .request_executor_slice();
      assert(ret.has_value()); // fresh executor -- should always have slices
                               // available
    }
    return ret;
  }

private:
  std::deque<aggregated_executor<Interface>> aggregation_executor_pool;
  std::atomic<size_t> current_interface{0};
  size_t slices_per_executor;
  aggregated_executor_modes mode;
  bool growing_pool{true};

private:
  /// Required for dealing with adding elements to the deque of
  /// aggregated_executors
  aggregation_mutex_t pool_mutex;
  /// Global access instance
  static std::unique_ptr<aggregation_pool[]>& instance(void) {
    static std::unique_ptr<aggregation_pool[]> pool_instances{
        new aggregation_pool[cppuddle::max_number_gpus]};
    return pool_instances;
  }
  static inline size_t number_devices = 1;
  static inline bool is_initialized = false;
  aggregation_pool() = default;

public:
  ~aggregation_pool() = default;
  // Bunch of constructors we don't need
  aggregation_pool(aggregation_pool const &other) = delete;
  aggregation_pool &operator=(aggregation_pool const &other) = delete;
  aggregation_pool(aggregation_pool &&other) = delete;
  aggregation_pool &operator=(aggregation_pool &&other) = delete;
};

} // namespace detail
} // namespace kernel_aggregation
} // namespace cppuddle

#endif
