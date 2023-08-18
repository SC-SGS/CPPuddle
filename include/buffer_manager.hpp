// Copyright (c) 2020-2023 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BUFFER_MANAGER_HPP
#define BUFFER_MANAGER_HPP

#include <atomic>
#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

// Warn about suboptimal performance without correct HPX-aware allocators
#ifdef CPPUDDLE_HAVE_HPX
#ifndef CPPUDDLE_HAVE_HPX_AWARE_ALLOCATORS
#pragma message                                                                \
"Warning: CPPuddle build with HPX support but without HPX-aware allocators enabled. \
For better performance configure CPPuddle with CPPUDDLE_WITH_HPX_AWARE_ALLOCATORS=ON!"
#else
// include runtime to get HPX thread IDs required for the HPX-aware allocators
#include <hpx/include/runtime.hpp>
#endif
#endif

#if defined(CPPUDDLE_HAVE_HPX) && defined(CPPUDDLE_HAVE_HPX_MUTEX)
// For builds with The HPX mutex
#include <hpx/mutex.hpp>
#endif

#ifdef CPPUDDLE_HAVE_COUNTERS
#include <boost/core/demangle.hpp>
#if defined(CPPUDDLE_HAVE_HPX)
#include <hpx/include/performance_counters.hpp>
#endif
#endif

#include "../include/detail/config.hpp"

namespace recycler {

namespace device_selection {
template <typename T, typename Allocator> struct select_device_functor {
  void operator()(const size_t device_id) {
    if constexpr (max_number_gpus > 1)
      throw std::runtime_error(
          "Allocators used in Multi-GPU builds need explicit Multi-GPU support "
          "(by having a select_device_functor overload");
  }
};
template <typename T> struct select_device_functor<T, std::allocator<T>> {
  void operator()(const size_t device_id) {}
};
} // namespace device_selection

namespace detail {


class buffer_recycler {
public:
#if defined(CPPUDDLE_DEACTIVATE_BUFFER_RECYCLING)

// Warn about suboptimal performance without recycling
#pragma message                                                                \
"Warning: Building without buffer recycling! Use only for performance testing! \
For better performance configure CPPuddle with CPPUDDLE_DEACTIVATE_BUFFER_RECYCLING=OFF!"

  template <typename T, typename Host_Allocator>
  static T *get(size_t number_elements, bool manage_content_lifetime = false,
      std::optional<size_t> location_hint = std::nullopt, 
      std::optional<size_t> device_id = std::nullopt) {

    return Host_Allocator{}.allocate(number_elements);
  }
  /// Marks an buffer as unused and fit for reusage
  template <typename T, typename Host_Allocator>
  static void mark_unused(T *p, size_t number_elements,
      std::optional<size_t> location_hint = std::nullopt,
      std::optional<size_t> device_id = std::nullopt) {
    return Host_Allocator{}.deallocate(p, number_elements);
  }
#else
  /// Returns and allocated buffer of the requested size - this may be a reused
  /// buffer
  template <typename T, typename Host_Allocator>
  static T *get(size_t number_elements, bool manage_content_lifetime = false,
      std::optional<size_t> location_hint = std::nullopt, 
      std::optional<size_t> device_id = std::nullopt) {
    return buffer_manager<T, Host_Allocator>::get(
        number_elements, manage_content_lifetime, location_hint, device_id);
  }
  /// Marks an buffer as unused and fit for reusage
  template <typename T, typename Host_Allocator>
  static void mark_unused(T *p, size_t number_elements,
      std::optional<size_t> location_hint = std::nullopt, 
      std::optional<size_t> device_id = std::nullopt) {
    return buffer_manager<T, Host_Allocator>::mark_unused(p, number_elements,
        location_hint, device_id);
  }
#endif
  template <typename T, typename Host_Allocator>
    static void register_allocator_counters_with_hpx(void) {
#ifdef CPPUDDLE_HAVE_COUNTERS
      buffer_manager<T, Host_Allocator>::register_counters_with_hpx();
#else
    std::cerr << "Warning: Trying to register allocator performance counters "
                 "with HPX but CPPuddle was built "
                 "without CPPUDDLE_WITH_COUNTERS -- operation will be ignored!"
              << std::endl;
#endif
  }

  /// Deallocate all buffers, no matter whether they are marked as used or not
  static void clean_all() {
    std::lock_guard<mutex_t> guard(instance().callback_protection_mut);
    for (const auto &clean_function :
         instance().total_cleanup_callbacks) {
      clean_function();
    }
  }
  /// Deallocated all currently unused buffer
  static void clean_unused_buffers() {
    std::lock_guard<mutex_t> guard(instance().callback_protection_mut);
    for (const auto &clean_function :
         instance().partial_cleanup_callbacks) {
      clean_function();
    }
  }
  /// Deallocate all buffers, no matter whether they are marked as used or not
  static void finalize() {
    std::lock_guard<mutex_t> guard(instance().callback_protection_mut);
    for (const auto &finalize_function :
         instance().finalize_callbacks) {
      finalize_function();
    }
  }

  static void print_performance_counters() {
#ifdef CPPUDDLE_HAVE_COUNTERS
    std::lock_guard<mutex_t> guard(instance().callback_protection_mut);
    for (const auto &print_function :
         instance().print_callbacks) {
      print_function();
    }
#else
    std::cerr << "Warning: Trying to print allocator performance counters but CPPuddle was built "
                 "without CPPUDDLE_WITH_COUNTERS -- operation will be ignored!"
              << std::endl;
#endif
  }

  // Member variables and methods
private:

  /// Singleton instance access
  static buffer_recycler& instance() {
    static buffer_recycler singleton{};
    return singleton;
  }
  /// Callbacks for printing the performance counter data
  std::list<std::function<void()>> print_callbacks;
  /// Callbacks for buffer_manager finalize - each callback completely destroys
  /// one buffer_manager
  std::list<std::function<void()>> finalize_callbacks;
  /// Callbacks for buffer_manager cleanups - each callback destroys all buffers within 
  /// one buffer_manager, both used and unsued
  std::list<std::function<void()>> total_cleanup_callbacks;
  /// Callbacks for partial buffer_manager cleanups - each callback deallocates
  /// all unused buffers of a manager
  std::list<std::function<void()>> partial_cleanup_callbacks;
  /// default, private constructor - not automatically constructed due to the
  /// deleted constructors
  buffer_recycler() = default;

  mutex_t callback_protection_mut;
  /// Add a callback function that gets executed upon cleanup and destruction
  static void add_total_cleanup_callback(const std::function<void()> &func) {
    std::lock_guard<mutex_t> guard(instance().callback_protection_mut);
    instance().total_cleanup_callbacks.push_back(func);
  }
  /// Add a callback function that gets executed upon partial (unused memory)
  /// cleanup
  static void add_partial_cleanup_callback(const std::function<void()> &func) {
    std::lock_guard<mutex_t> guard(instance().callback_protection_mut);
    instance().partial_cleanup_callbacks.push_back(func);
  }
  /// Add a callback function that gets executed upon partial (unused memory)
  /// cleanup
  static void add_finalize_callback(const std::function<void()> &func) {
    std::lock_guard<mutex_t> guard(instance().callback_protection_mut);
    instance().finalize_callbacks.push_back(func);
  }
  /// Add a callback function that gets executed upon partial (unused memory)
  /// cleanup
  static void add_print_callback(const std::function<void()> &func) {
    std::lock_guard<mutex_t> guard(instance().callback_protection_mut);
    instance().print_callbacks.push_back(func);
  }

public:
  ~buffer_recycler() = default; 

  // Subclasses
private:
  /// Memory Manager subclass to handle buffers a specific type
  template <typename T, typename Host_Allocator> class buffer_manager {
  private:
    // Tuple content: Pointer to buffer, buffer_size, location ID, Flag
    // The flag at the end controls whether to buffer content is to be reused as
    // well
    using buffer_entry_type = std::tuple<T *, size_t, size_t, bool>;


  public:
    /// Cleanup and delete this singleton
    static void clean() {
      assert(instance() && !is_finalized);
      for (auto i = 0; i < number_instances * max_number_gpus; i++) {
        std::lock_guard<mutex_t> guard(instance()[i].mut);
        instance()[i].clean_all_buffers();
      }
    }
    static void print_performance_counters() {
      assert(instance() && !is_finalized);
      for (auto i = 0; i < number_instances * max_number_gpus; i++) {
        std::lock_guard<mutex_t> guard(instance()[i].mut);
        instance()[i].print_counters();
      }
    }
    static void finalize() {
      assert(instance() && !is_finalized);
      is_finalized = true;
      for (auto i = 0; i < number_instances * max_number_gpus; i++) {
        std::lock_guard<mutex_t> guard(instance()[i].mut);
        instance()[i].clean_all_buffers();
      }
      instance().reset();
    }
    /// Cleanup all buffers not currently in use
    static void clean_unused_buffers_only() {
      assert(instance() && !is_finalized);
      for (auto i = 0; i < number_instances * max_number_gpus; i++) {
        std::lock_guard<mutex_t> guard(instance()[i].mut);
        for (auto &buffer_tuple : instance()[i].unused_buffer_list) {
          Host_Allocator alloc;
          if (std::get<3>(buffer_tuple)) {
            std::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
          }
          alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        instance()[i].unused_buffer_list.clear();
      }
    }
#if defined(CPPUDDLE_HAVE_COUNTERS) && defined(CPPUDDLE_HAVE_HPX)
    static size_t get_sum_number_recycling(bool reset) {
      if (reset)
        sum_number_recycling = 0;
      return sum_number_recycling;
    }
    static size_t get_sum_number_allocation(bool reset) {
      if (reset)
        sum_number_allocation = 0;
      return sum_number_allocation;
    }
    static size_t get_sum_number_creation(bool reset) {
      if (reset)
        sum_number_creation = 0;
      return sum_number_creation;
    }
    static size_t get_sum_number_deallocation(bool reset) {
      if (reset)
        sum_number_deallocation = 0;
      return sum_number_deallocation;
    }
    static size_t get_sum_number_wrong_hints(bool reset) {
      if (reset)
        sum_number_wrong_hints = 0;
      return sum_number_wrong_hints;
    }
    static size_t get_sum_number_wrong_device_hints(bool reset) {
      if (reset)
        sum_number_wrong_hints = 0;
      return sum_number_wrong_device_hints;
    }
    static size_t get_sum_number_bad_allocs(bool reset) {
      if (reset)
        sum_number_bad_allocs = 0;
      return sum_number_bad_allocs;
    }

    static void register_counters_with_hpx(void) {
      std::string alloc_name =
          boost::core::demangle(typeid(Host_Allocator).name()) +
          std::string("_") + boost::core::demangle(typeid(T).name());
      hpx::performance_counters::install_counter_type(
          std::string("/cppuddle/allocators/") + alloc_name + std::string("/number_recycling/"),
          &get_sum_number_recycling,
          "Number of allocations using a recycled buffer with this "
          "allocator");
      hpx::performance_counters::install_counter_type(
          std::string("/cppuddle/allocators/") + alloc_name + std::string("/number_allocations/"),
          &get_sum_number_allocation,
          "Number of allocations with this allocator");
      hpx::performance_counters::install_counter_type(
          std::string("/cppuddle/allocators/") + alloc_name + std::string("/number_creations/"),
          &get_sum_number_creation,
          "Number of allocations not using a recycled buffer with this "
          "allocator");
      hpx::performance_counters::install_counter_type(
          std::string("/cppuddle/allocators/") + alloc_name + std::string("/number_deallocations/"),
          &get_sum_number_deallocation,
          "Number of deallocations yielding buffers to be recycled with this "
          "allocator");
      hpx::performance_counters::install_counter_type(
          std::string("/cppuddle/allocators/") + alloc_name + std::string("/number_wrong_hints/"),
          &get_sum_number_wrong_hints,
          "Number of wrong hints supplied to the dealloc method with this allocator");
      hpx::performance_counters::install_counter_type(
          std::string("/cppuddle/allocators/") + alloc_name + std::string("/number_wrong_device_hints/"),
          &get_sum_number_wrong_device_hints,
          "Number of wrong device hints supplied to the dealloc method with this allocator");
      hpx::performance_counters::install_counter_type(
          std::string("/cppuddle/allocators/") + alloc_name + std::string("/number_bad_allocs/"),
          &get_sum_number_bad_allocs,
          "Number of wrong bad allocs which triggered a cleanup of unused buffers");
    }
#endif

    /// Tries to recycle or create a buffer of type T and size number_elements.
    static T *get(size_t number_of_elements, bool manage_content_lifetime,
        std::optional<size_t> location_hint = std::nullopt,
        std::optional<size_t> gpu_device_id = std::nullopt) {
      init_callbacks_once();
      if (is_finalized) {
        throw std::runtime_error("Tried allocation after finalization");
      }
      assert(instance() && !is_finalized);

      size_t location_id = 0;
      if (location_hint) {
        location_id = *location_hint;
      }
      if (location_id >= number_instances) {
        throw std::runtime_error("Tried to create buffer with invalid location_id [get]");
      }
      size_t device_id = 0;
      if (gpu_device_id) {
        device_id = *gpu_device_id;
      }
      if (device_id >= max_number_gpus) {
        throw std::runtime_error("Tried to create buffer with invalid device id [get]! "
                                 "Is multigpu support enabled with the correct number "
                                 "of GPUs?");
      }

      location_id = location_id + device_id * number_instances;
      std::lock_guard<mutex_t> guard(instance()[location_id].mut);


#ifdef CPPUDDLE_HAVE_COUNTERS
      instance()[location_id].number_allocation++;
      sum_number_allocation++;
#endif
      // Check for unused buffers we can recycle:
      for (auto iter = instance()[location_id].unused_buffer_list.begin();
           iter != instance()[location_id].unused_buffer_list.end(); iter++) {
        auto tuple = *iter;
        if (std::get<1>(tuple) == number_of_elements) {
          instance()[location_id].unused_buffer_list.erase(iter);

          // handle the switch from aggressive to non aggressive reusage (or
          // vice-versa)
          if (manage_content_lifetime && !std::get<3>(tuple)) {
            std::uninitialized_value_construct_n(std::get<0>(tuple),
                                                  number_of_elements);
            std::get<3>(tuple) = true;
          } else if (!manage_content_lifetime && std::get<3>(tuple)) {
            std::destroy_n(std::get<0>(tuple), std::get<1>(tuple));
            std::get<3>(tuple) = false;
          }
          instance()[location_id].buffer_map.insert({std::get<0>(tuple), tuple});
#ifdef CPPUDDLE_HAVE_COUNTERS
          instance()[location_id].number_recycling++;
          sum_number_recycling++;
#endif
          return std::get<0>(tuple);
        }
      }

      // No unused buffer found -> Create new one and return it
      try {
        recycler::device_selection::select_device_functor<T, Host_Allocator>{}(
            device_id);
        Host_Allocator alloc;
        T *buffer = alloc.allocate(number_of_elements);
        instance()[location_id].buffer_map.insert(
            {buffer, std::make_tuple(buffer, number_of_elements, 1,
                                     manage_content_lifetime)});
#ifdef CPPUDDLE_HAVE_COUNTERS
        instance()[location_id].number_creation++;
        sum_number_creation++;
#endif
        if (manage_content_lifetime) {
          std::uninitialized_value_construct_n(buffer, number_of_elements);
        }
        return buffer;
      } catch (std::bad_alloc &e) {
        // not enough memory left! Cleanup and attempt again:
        std::cerr 
          << "Not enough memory left. Cleaning up unused buffers now..." 
          << std::endl;
        buffer_recycler::clean_unused_buffers();
        std::cerr << "Buffers cleaned! Try allocation again..." << std::endl;

        // If there still isn't enough memory left, the caller has to handle it
        // We've done all we can in here
        Host_Allocator alloc;
        recycler::device_selection::select_device_functor<T, Host_Allocator>{}(
            device_id);
        T *buffer = alloc.allocate(number_of_elements);
        instance()[location_id].buffer_map.insert(
            {buffer, std::make_tuple(buffer, number_of_elements, 1,
                                     manage_content_lifetime)});
#ifdef CPPUDDLE_HAVE_COUNTERS
        instance()[location_id].number_creation++;
        sum_number_creation++;
        instance()[location_id].number_bad_alloc++;
        sum_number_bad_allocs++;
#endif
        std::cerr << "Second attempt allocation successful!" << std::endl;
        if (manage_content_lifetime) {
          std::uninitialized_value_construct_n(buffer, number_of_elements);
        }
        return buffer;
      }
    }

    static void mark_unused(T *memory_location, size_t number_of_elements,
        std::optional<size_t> location_hint = std::nullopt,
        std::optional<size_t> device_hint = std::nullopt) {
      if (is_finalized)
        return;
      assert(instance() && !is_finalized);

      size_t location_id = 0;
      if (location_hint) {
        location_id = *location_hint;
        if (location_id >= number_instances) {
          throw std::runtime_error(
              "Buffer recylcer received invalid location hint [mark_unused]");
        }
      }
      size_t device_id = 0;
      if (device_hint) {
        device_id = *device_hint;
        if (device_id >= max_number_gpus) {
          throw std::runtime_error(
              "Buffer recylcer received invalid devce hint [mark_unused]");
        }
      }

      // Attempt 1 to find the correct bucket/location: Look at provided hint:
      if (location_hint) {
        size_t location_id = location_hint.value() + device_id * number_instances;
        std::lock_guard<mutex_t> guard(instance()[location_id].mut);
        if (instance()[location_id].buffer_map.find(memory_location) !=
            instance()[location_id].buffer_map.end()) {
#ifdef CPPUDDLE_HAVE_COUNTERS
          instance()[location_id].number_deallocation++;
          sum_number_deallocation++;
#endif
          auto it = instance()[location_id].buffer_map.find(memory_location);
          assert(it != instance()[location_id].buffer_map.end());
          auto &tuple = it->second;
          // sanity checks:
          assert(std::get<1>(tuple) == number_of_elements);
          // move to the unused_buffer list
          instance()[location_id].unused_buffer_list.push_front(tuple);
          instance()[location_id].buffer_map.erase(memory_location);
          return; // Success
        }
        // hint was wrong 
#ifdef CPPUDDLE_HAVE_COUNTERS
        instance()[location_id].number_wrong_hints++;
        sum_number_wrong_hints++;
#endif
      }
      // Failed to find buffer in the specified localtion/device!
      // Attempt 2 - Look for buffer other locations on the same device...
      for (size_t location_id = device_id * number_instances;
           location_id < (device_id + 1) * number_instances; location_id++) {
        if (location_hint) {
          if (*location_hint + device_id * max_number_gpus  == location_id) {
            continue; // already tried this -> skip
          }
        }
        std::lock_guard<mutex_t> guard(instance()[location_id].mut);
        if (instance()[location_id].buffer_map.find(memory_location) !=
            instance()[location_id].buffer_map.end()) {
#ifdef CPPUDDLE_HAVE_COUNTERS
          instance()[location_id].number_deallocation++;
          sum_number_deallocation++;
#endif
          auto it = instance()[location_id].buffer_map.find(memory_location);
          assert(it != instance()[location_id].buffer_map.end());
          auto &tuple = it->second;
          // sanity checks:
          assert(std::get<1>(tuple) == number_of_elements);
          // move to the unused_buffer list
          instance()[location_id].unused_buffer_list.push_front(tuple);
          instance()[location_id].buffer_map.erase(memory_location);
          return; // Success
        }
      }
      // device hint was wrong 
#ifdef CPPUDDLE_HAVE_COUNTERS
      if (device_hint) {
        sum_number_wrong_device_hints++;
      }
#endif
      // Failed to find buffer on the specified device!
      // Attempt 3 - Look for buffer on other devices...
      for (size_t local_device_id = 0; local_device_id < max_number_gpus;
           local_device_id++) {
        if (local_device_id == device_id)
          continue; // aldready tried this device

        // Try hint localtion first yet again (though on different device)
        if (location_hint) {
          size_t location_id = location_hint.value() + local_device_id * number_instances;
          std::lock_guard<mutex_t> guard(instance()[location_id].mut);
          if (instance()[location_id].buffer_map.find(memory_location) !=
              instance()[location_id].buffer_map.end()) {
#ifdef CPPUDDLE_HAVE_COUNTERS
            instance()[location_id].number_deallocation++;
            sum_number_deallocation++;
#endif
            auto it = instance()[location_id].buffer_map.find(memory_location);
            assert(it != instance()[location_id].buffer_map.end());
            auto &tuple = it->second;
            // sanity checks:
            assert(std::get<1>(tuple) == number_of_elements);
            // move to the unused_buffer list
            instance()[location_id].unused_buffer_list.push_front(tuple);
            instance()[location_id].buffer_map.erase(memory_location);
            return; // Success
          }
        }
        // Failed - check all other localtions on device
        for (size_t location_id = local_device_id * number_instances;
             location_id < (local_device_id + 1) * number_instances; location_id++) {
          if (location_hint) {
            if (*location_hint + local_device_id * max_number_gpus == location_id) {
              continue; // already tried this -> skip
            }
          }
          std::lock_guard<mutex_t> guard(instance()[location_id].mut);
          if (instance()[location_id].buffer_map.find(memory_location) !=
              instance()[location_id].buffer_map.end()) {
#ifdef CPPUDDLE_HAVE_COUNTERS
            instance()[location_id].number_deallocation++;
            sum_number_deallocation++;
#endif
            auto it = instance()[location_id].buffer_map.find(memory_location);
            assert(it != instance()[location_id].buffer_map.end());
            auto &tuple = it->second;
            // sanity checks:
            assert(std::get<1>(tuple) == number_of_elements);
            // move to the unused_buffer list
            instance()[location_id].unused_buffer_list.push_front(tuple);
            instance()[location_id].buffer_map.erase(memory_location);
            return; // Success
          }
        }
      }
      // Buffer that is to be deleted is nowhere to be found - we looked everywhere!
      // =>
      // Failure! Handle here...

      // TODO Throw exception instead in the futures, as soon as the recycler finalize is 
      // in all user codes
      /* throw std::runtime_error("Tried to delete non-existing buffer"); */

      // This is odd: Print warning -- however, might also happen with static
      // buffers using these allocators IF the new finalize was not called. For
      // now, print warning until all user-code is upgraded to the finalize method.
      // This allows using current versions of cppuddle with older application code
      std::cerr
          << "Warning! Tried to delete non-existing buffer within CPPuddle!"
          << std::endl;
      std::cerr << "Did you forget to call recycler::finalize?" << std::endl;
      }

  private:
    /// List with all buffers still in usage
    std::unordered_map<T *, buffer_entry_type> buffer_map{};
    /// List with all buffers currently not used
    std::list<buffer_entry_type> unused_buffer_list{};
    /// Access control
    mutex_t mut;
#ifdef CPPUDDLE_HAVE_COUNTERS
    /// Performance counters
    size_t number_allocation{0}, number_deallocation{0}, number_wrong_hints{0},
        number_recycling{0}, number_creation{0}, number_bad_alloc{0};

    static inline std::atomic<size_t> sum_number_allocation{0},
        sum_number_deallocation{0}, sum_number_wrong_hints{0},
        sum_number_wrong_device_hints{0}, sum_number_recycling{0},
        sum_number_creation{0}, sum_number_bad_allocs{0};
#endif
        /// default, private constructor - not automatically constructed due to
        /// the deleted constructors
        buffer_manager() = default;
    buffer_manager&
    operator=(buffer_manager<T, Host_Allocator> const &other) = default;
    buffer_manager&
    operator=(buffer_manager<T, Host_Allocator> &&other) = delete;
    static std::unique_ptr<buffer_manager[]>& instance(void) {
      static std::unique_ptr<buffer_manager[]> instances{
          new buffer_manager[number_instances * max_number_gpus]};
      return instances;
    }
    static void init_callbacks_once(void) {
      assert(instance());
#if defined(CPPUDDLE_HAVE_HPX)  && defined(CPPUDDLE_HAVE_HPX_MUTEX)
      static hpx::once_flag flag; 
      hpx::call_once(flag, []() {
#else
      static std::once_flag flag; 
      std::call_once(flag, []() {
#endif
        is_finalized = false;
        buffer_recycler::add_total_cleanup_callback(clean);
        buffer_recycler::add_partial_cleanup_callback(
            clean_unused_buffers_only);
        buffer_recycler::add_finalize_callback(
            finalize);
#ifdef CPPUDDLE_HAVE_COUNTERS
        buffer_recycler::add_print_callback(
            print_performance_counters);
#endif
          });
    }
    static inline std::atomic<bool> is_finalized;

#ifdef CPPUDDLE_HAVE_COUNTERS
    void print_counters(void) {
      if (number_allocation == 0)
        return;
      // Print performance counters
      size_t number_cleaned = unused_buffer_list.size() + buffer_map.size();
      std::cout << "\nBuffer manager destructor for (Alloc: "
                << boost::core::demangle(typeid(Host_Allocator).name()) << ", Type: "
                << boost::core::demangle(typeid(T).name())
                << "):" << std::endl
                << "--------------------------------------------------------------------"
                << std::endl
                << "--> Number of bad_allocs that triggered garbage "
                   "collection:       "
                << number_bad_alloc << std::endl
                << "--> Number of buffers that got requested from this "
                   "manager:       "
                << number_allocation << std::endl
                << "--> Number of times an unused buffer got recycled for a "
                   "request:  "
                << number_recycling << std::endl
                << "--> Number of times a new buffer had to be created for a "
                   "request: "
                << number_creation << std::endl
                << "--> Number cleaned up buffers:                             "
                   "       "
                << number_cleaned << std::endl
                << "--> Number wrong deallocation hints:                       "
                   "       "
                << number_wrong_hints << std::endl
                << "--> Number of buffers that were marked as used upon "
                   "cleanup:      "
                << buffer_map.size() << std::endl
                << "==> Recycle rate:                                          "
                   "       "
                << static_cast<float>(number_recycling) / number_allocation *
                       100.0f
                << "%" << std::endl;
    }
#endif

    void clean_all_buffers(void) {
#ifdef CPPUDDLE_HAVE_COUNTERS
      if (number_allocation == 0 && number_recycling == 0 &&
          number_bad_alloc == 0 && number_creation == 0 &&
          unused_buffer_list.empty() && buffer_map.empty()) {
        return;
      }
#endif
      for (auto &buffer_tuple : unused_buffer_list) {
        Host_Allocator alloc;
        if (std::get<3>(buffer_tuple)) {
          std::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
      }
      for (auto &map_tuple : buffer_map) {
        auto buffer_tuple = map_tuple.second;
        Host_Allocator alloc;
        if (std::get<3>(buffer_tuple)) {
          std::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
      }
      unused_buffer_list.clear();
      buffer_map.clear();
#ifdef CPPUDDLE_HAVE_COUNTERS
      number_allocation = 0;
      number_recycling = 0;
      number_bad_alloc = 0;
      number_creation = 0;
      number_wrong_hints = 0;
#endif
    }
  public:
    ~buffer_manager() {
      clean_all_buffers();
    }

  public: // Putting deleted constructors in public gives more useful error
          // messages
    // Bunch of constructors we don't need
    buffer_manager(
        buffer_manager<T, Host_Allocator> const &other) = delete;
    buffer_manager(
        buffer_manager<T, Host_Allocator> &&other) = delete;
  };

public:
  // Putting deleted constructors in public gives more useful error messages
  // Bunch of constructors we don't need
  buffer_recycler(buffer_recycler const &other) = delete;
  buffer_recycler& operator=(buffer_recycler const &other) = delete;
  buffer_recycler(buffer_recycler &&other) = delete;
  buffer_recycler& operator=(buffer_recycler &&other) = delete;
};

template <typename T, typename Host_Allocator> struct recycle_allocator {
  using value_type = T;
  using underlying_allocator_type = Host_Allocator;
  static_assert(std::is_same_v<value_type, typename underlying_allocator_type::value_type>);
  const std::optional<size_t> dealloc_hint;
  const std::optional<size_t> device_id;

#ifndef CPPUDDLE_HAVE_HPX_AWARE_ALLOCATORS
  recycle_allocator() noexcept
      : dealloc_hint(std::nullopt), device_id(std::nullopt) {}
  explicit recycle_allocator(size_t hint) noexcept
      : dealloc_hint(std::nullopt), device_id(std::nullopt) {}
  explicit recycle_allocator(
      recycle_allocator<T, Host_Allocator> const &other) noexcept
      : dealloc_hint(std::nullopt), device_id(std::nullopt) {}
  T *allocate(std::size_t n) {
    T *data = buffer_recycler::get<T, Host_Allocator>(n);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n);
  }
#else
  recycle_allocator() noexcept
      : dealloc_hint(hpx::get_worker_thread_num()), device_id(0) {}
  explicit recycle_allocator(const size_t device_id) noexcept
      : dealloc_hint(hpx::get_worker_thread_num()), device_id(device_id) {}
  explicit recycle_allocator(const size_t device_i, const size_t location_id) noexcept
      : dealloc_hint(location_id), device_id(device_id) {}
  explicit recycle_allocator(
      recycle_allocator<T, Host_Allocator> const &other) noexcept
  : dealloc_hint(other.dealloc_hint), device_id(other.device_id) {}
  T *allocate(std::size_t n) {
    T *data = buffer_recycler::get<T, Host_Allocator>(
        n, false, hpx::get_worker_thread_num(), device_id);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n, dealloc_hint,
                                                    device_id);
  }
#endif

  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    ::new (static_cast<void *>(p)) T(std::forward<Args>(args)...);
  }
  void destroy(T *p) { p->~T(); }
};
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator==(recycle_allocator<T, Host_Allocator> const &,
           recycle_allocator<U, Host_Allocator> const &) noexcept {
  if constexpr (std::is_same_v<T, U>)
    return true;
  else 
    return false;
}
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator!=(recycle_allocator<T, Host_Allocator> const &,
           recycle_allocator<U, Host_Allocator> const &) noexcept {
  if constexpr (std::is_same_v<T, U>)
    return false;
  else 
    return true;
}

/// Recycles not only allocations but also the contents of a buffer
template <typename T, typename Host_Allocator>
struct aggressive_recycle_allocator {
  using value_type = T;
  using underlying_allocator_type = Host_Allocator;
  static_assert(std::is_same_v<value_type, typename underlying_allocator_type::value_type>);
  const std::optional<size_t> dealloc_hint;
  const std::optional<size_t> device_id;

#ifndef CPPUDDLE_HAVE_HPX_AWARE_ALLOCATORS
  aggressive_recycle_allocator() noexcept
      : dealloc_hint(std::nullopt), device_id(std::nullopt) {}
  explicit aggressive_recycle_allocator(size_t hint) noexcept
      : dealloc_hint(std::nullopt), device_id(std::nullopt) {}
  explicit aggressive_recycle_allocator(
      aggressive_recycle_allocator<T, Host_Allocator> const &) noexcept 
  : dealloc_hint(std::nullopt), device_id(std::nullopt) {}
  T *allocate(std::size_t n) {
    T *data = buffer_recycler::get<T, Host_Allocator>(
        n, true); // also initializes the buffer if it isn't reused
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n);
  }
#else
  aggressive_recycle_allocator() noexcept
      : dealloc_hint(hpx::get_worker_thread_num()), device_id(0) {}
  explicit aggressive_recycle_allocator(const size_t device_id) noexcept
      : dealloc_hint(hpx::get_worker_thread_num()), device_id(device_id) {}
  explicit aggressive_recycle_allocator(const size_t device_id, const size_t location_id) noexcept
      : dealloc_hint(location_id), device_id(device_id) {}
  explicit aggressive_recycle_allocator(
      recycle_allocator<T, Host_Allocator> const &other) noexcept 
    : dealloc_hint(other.dealloc_hint), device_id(other.device_id) {}
  T *allocate(std::size_t n) {
    T *data = buffer_recycler::get<T, Host_Allocator>(
        n, true, dealloc_hint, device_id); // also initializes the buffer
                                                // if it isn't reused
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n, dealloc_hint,
                                                    device_id);
  }
#endif

#ifndef CPPUDDLE_DEACTIVATE_AGGRESSIVE_ALLOCATORS
  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    // Do nothing here - we reuse the content of the last owner
  }
  void destroy(T *p) {
    // Do nothing here - Contents will be destroyed when the buffer manager is
    // destroyed, not before
  }
#else
// Warn about suboptimal performance without recycling
#pragma message                                                                \
"Warning: Building without content reusage for aggressive allocators! \
For better performance configure with CPPUDDLE_DEACTIVATE_AGGRESSIVE_ALLOCATORS=OFF !"
  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    ::new (static_cast<void *>(p)) T(std::forward<Args>(args)...);
  }
  void destroy(T *p) { p->~T(); }
#endif
};

template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator==(aggressive_recycle_allocator<T, Host_Allocator> const &,
           aggressive_recycle_allocator<U, Host_Allocator> const &) noexcept {
  if constexpr (std::is_same_v<T, U>)
    return true;
  else 
    return false;
}
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator!=(aggressive_recycle_allocator<T, Host_Allocator> const &,
           aggressive_recycle_allocator<U, Host_Allocator> const &) noexcept {
  if constexpr (std::is_same_v<T, U>)
    return false;
  else 
    return true;
}

} // namespace detail

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_std = detail::recycle_allocator<T, std::allocator<T>>;
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_std =
    detail::aggressive_recycle_allocator<T, std::allocator<T>>;

inline void print_performance_counters() { detail::buffer_recycler::print_performance_counters(); }
/// Deletes all buffers (even ones still marked as used), delete the buffer
/// managers and the recycler itself
inline void force_cleanup() { detail::buffer_recycler::clean_all(); }
/// Deletes all buffers currently marked as unused
inline void cleanup() { detail::buffer_recycler::clean_unused_buffers(); }
/// Deletes all buffers (even ones still marked as used), delete the buffer
/// managers and the recycler itself. Disallows further usage.
inline void finalize() { detail::buffer_recycler::finalize(); }

} // end namespace recycler

#endif
