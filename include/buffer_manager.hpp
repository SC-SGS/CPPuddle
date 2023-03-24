// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BUFFER_MANAGER_HPP
#define BUFFER_MANAGER_HPP

#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#ifdef CPPUDDLE_HAVE_COUNTERS
#include <boost/core/demangle.hpp>
#endif

namespace recycler {
namespace detail {

namespace util {
/// Helper methods for C++14 - this is obsolete for c++17 and only meant as a
/// temporary crutch
template <typename ForwardIt, typename Size>
void uninitialized_value_construct_n(ForwardIt first, Size n) {
  using Value = typename std::iterator_traits<ForwardIt>::value_type;
  ForwardIt current = first;
  for (; n > 0; (void)++current, --n) {
    ::new (static_cast<void *>(std::addressof(*current))) Value();
  }
}
/// Helper methods for C++14 - this is obsolete for c++17 and only meant as a
/// temporary crutch
template <typename ForwardIt, typename Size>
void destroy_n(ForwardIt first, Size n) {
  using Value = typename std::iterator_traits<ForwardIt>::value_type;
  ForwardIt current = first;
  for (; n > 0; (void)++current, --n) {
    current->~Value();
  }
}
} // namespace util

class buffer_recycler {
  // Public interface
public:
  /// Returns and allocated buffer of the requested size - this may be a reused
  /// buffer
  template <typename T, typename Host_Allocator>
  static T *get(size_t number_elements, bool manage_content_lifetime = false) {
    std::lock_guard<std::mutex> guard(mut);
    if (!recycler_instance) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      recycler_instance.reset(new buffer_recycler());
    }
    return buffer_manager<T, Host_Allocator>::get(number_elements,
                                                  manage_content_lifetime);
  }
  /// Marks an buffer as unused and fit for reusage
  template <typename T, typename Host_Allocator>
  static void mark_unused(T *p, size_t number_elements) {
    std::lock_guard<std::mutex> guard(mut);
    if (recycler_instance) { // if the instance was already destroyed all
                             // buffers are destroyed anyway
      return buffer_manager<T, Host_Allocator>::mark_unused(p, number_elements);
    }
  }
  /// Increase the reference coutner of a buffer
  template <typename T, typename Host_Allocator>
  static void increase_usage_counter(T *p, size_t number_elements) noexcept {
    std::lock_guard<std::mutex> guard(mut);
    if (recycler_instance) { // if the instance was already destroyed all
                             // buffers are destroyed anyway
      return buffer_manager<T, Host_Allocator>::increase_usage_counter(
          p, number_elements);
    }
  }
  /// Deallocate all buffers, no matter whether they are marked as used or not
  static void clean_all() {
    std::lock_guard<std::mutex> guard(mut);
    if (recycler_instance) {
      for (const auto &clean_function :
           recycler_instance->total_cleanup_callbacks) {
        clean_function();
      }
    }
    recycler_instance.reset();
  }
  /// Deallocated all currently unused buffer
  static void clean_unused_buffers() {
    std::lock_guard<std::mutex> guard(mut);
    if (recycler_instance) {
      for (const auto &clean_function :
           recycler_instance->partial_cleanup_callbacks) {
        clean_function();
      }
    }
  }

  // Member variables and methods
private:
  /// Singleton instance pointer
  static std::unique_ptr<buffer_recycler> recycler_instance;
  /// Callbacks for buffer_manager cleanups - each callback completely destroys
  /// one buffer_manager
  std::list<std::function<void()>> total_cleanup_callbacks;
  /// Callbacks for partial buffer_manager cleanups - each callback deallocates
  /// all unused buffers of a manager
  std::list<std::function<void()>> partial_cleanup_callbacks;
  /// One Mutex to control concurrent access - Since we do not actually ever
  /// return the singleton instance anywhere, this should hopefully suffice We
  /// want more fine-grained concurrent access eventually
  static std::mutex mut;
  /// default, private constructor - not automatically constructed due to the
  /// deleted constructors
  buffer_recycler() = default;
  /// Add a callback function that gets executed upon cleanup and destruction
  static void add_total_cleanup_callback(const std::function<void()> &func) {
    // This methods assumes instance is initialized since it is a private method
    // and all static public methods have guards
    recycler_instance->total_cleanup_callbacks.push_back(func);
  }
  /// Add a callback function that gets executed upon partial (unused memory)
  /// cleanup
  static void add_partial_cleanup_callback(const std::function<void()> &func) {
    // This methods assumes instance is initialized since it is a private method
    // and all static public methods have guards
    recycler_instance->partial_cleanup_callbacks.push_back(func);
  }

public:
  ~buffer_recycler() = default; // public destructor for unique_ptr instance

  // Subclasses
private:
  /// Memory Manager subclass to handle buffers a specific type
  template <typename T, typename Host_Allocator> class buffer_manager {
  private:
    // Tuple content: Pointer to buffer, buffer_size, reference_counter, Flag
    // The flag at the end controls whether to buffer content is to be reused as
    // well
    using buffer_entry_type = std::tuple<T *, size_t, size_t, bool>;

  public:
    /// Cleanup and delete this singleton
    static void clean() { manager_instance.reset(); }
    /// Cleanup all buffers not currently in use
    static void clean_unused_buffers_only() {
      if (!manager_instance) {
        return;
      }
      for (auto &buffer_tuple : manager_instance->unused_buffer_list) {
        Host_Allocator alloc;
        if (std::get<3>(buffer_tuple)) {
          util::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
      }
      manager_instance->unused_buffer_list.clear();
    }

    /// Tries to recycle or create a buffer of type T and size number_elements.
    static T *get(size_t number_of_elements, bool manage_content_lifetime) {
      if (!manager_instance) {
        manager_instance.reset(new buffer_manager());
        buffer_recycler::add_total_cleanup_callback(clean);
        buffer_recycler::add_partial_cleanup_callback(
            clean_unused_buffers_only);
      }
#ifdef CPPUDDLE_HAVE_COUNTERS
      manager_instance->number_allocation++;
#endif
      // Check for unused buffers we can recycle:
      for (auto iter = manager_instance->unused_buffer_list.begin();
           iter != manager_instance->unused_buffer_list.end(); iter++) {
        auto tuple = *iter;
        if (std::get<1>(tuple) == number_of_elements) {
          manager_instance->unused_buffer_list.erase(iter);
          std::get<2>(tuple)++; // increase usage counter to 1

          // handle the switch from aggressive to non aggressive reusage (or
          // vice-versa)
          if (manage_content_lifetime && !std::get<3>(tuple)) {
            util::uninitialized_value_construct_n(std::get<0>(tuple),
                                                  number_of_elements);
            std::get<3>(tuple) = true;
          } else if (!manage_content_lifetime && std::get<3>(tuple)) {
            util::destroy_n(std::get<0>(tuple), std::get<1>(tuple));
            std::get<3>(tuple) = false;
          }
          manager_instance->buffer_map.insert({std::get<0>(tuple), tuple});
#ifdef CPPUDDLE_HAVE_COUNTERS
          manager_instance->number_recycling++;
#endif
          return std::get<0>(tuple);
        }
      }

      // No unsued buffer found -> Create new one and return it
      try {
        Host_Allocator alloc;
        T *buffer = alloc.allocate(number_of_elements);
        manager_instance->buffer_map.insert(
            {buffer, std::make_tuple(buffer, number_of_elements, 1,
                                     manage_content_lifetime)});
#ifdef CPPUDDLE_HAVE_COUNTERS
        manager_instance->number_creation++;
#endif
        if (manage_content_lifetime) {
          util::uninitialized_value_construct_n(buffer, number_of_elements);
        }
        return buffer;
      } catch (std::bad_alloc &e) {
        // not enough memory left! Cleanup and attempt again:
        buffer_recycler::clean_unused_buffers();

        // If there still isn't enough memory left, the caller has to handle it
        // We've done all we can in here
        Host_Allocator alloc;
        T *buffer = alloc.allocate(number_of_elements);
        manager_instance->buffer_map.insert(
            {buffer, std::make_tuple(buffer, number_of_elements, 1,
                                     manage_content_lifetime)});
#ifdef CPPUDDLE_HAVE_COUNTERS
        manager_instance->number_creation++;
        manager_instance->number_bad_alloc++;
#endif
        if (manage_content_lifetime) {
          util::uninitialized_value_construct_n(buffer, number_of_elements);
        }
        return buffer;
      }
    }

    static void mark_unused(T *memory_location, size_t number_of_elements) {
      // This will never be called without an instance since all access for this
      // method comes from the buffer recycler We can forego the instance
      // existence check here
      assert(manager_instance);
#ifdef CPPUDDLE_HAVE_COUNTERS
      manager_instance->number_dealloacation++;
#endif
      auto it = manager_instance->buffer_map.find(memory_location);
      assert(it != manager_instance->buffer_map.end());
      auto &tuple = it->second;
      // sanity checks:
      assert(std::get<1>(tuple) == number_of_elements);
      assert(std::get<2>(tuple) >= 1);
      std::get<2>(tuple)--;          // decrease usage counter
      if (std::get<2>(tuple) == 0) { // not used anymore?
        // move to the unused_buffer list
        manager_instance->unused_buffer_list.push_front(tuple);
        manager_instance->buffer_map.erase(memory_location);
      }
    }

    static void increase_usage_counter(T *memory_location,
                                       size_t number_of_elements) noexcept {
      auto it = manager_instance->buffer_map.find(memory_location);
      assert(it != manager_instance->buffer_map.end());
      auto &tuple = it->second;
      // sanity checks:
      assert(std::get<1>(tuple) == number_of_elements);
      assert(std::get<2>(tuple) >= 1);
      std::get<2>(tuple)++; // increase usage counter
    }

  private:
    /// List with all buffers still in usage
    std::unordered_map<T *, buffer_entry_type> buffer_map{};
    /// List with all buffers currently not used
    std::list<buffer_entry_type> unused_buffer_list{};
#ifdef CPPUDDLE_HAVE_COUNTERS
    /// Performance counters
    size_t number_allocation{0}, number_dealloacation{0};
    size_t number_recycling{0}, number_creation{0}, number_bad_alloc{0};
#endif
    /// Singleton instance
    static std::unique_ptr<buffer_manager<T, Host_Allocator>> manager_instance;
    /// default, private constructor - not automatically constructed due to the
    /// deleted constructors
    buffer_manager() = default;

  public:
    ~buffer_manager() {
      for (auto &buffer_tuple : unused_buffer_list) {
        Host_Allocator alloc;
        if (std::get<3>(buffer_tuple)) {
          util::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
      }
      for (auto &map_tuple : buffer_map) {
        auto buffer_tuple = map_tuple.second;
        Host_Allocator alloc;
        if (std::get<3>(buffer_tuple)) {
          util::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
      }
#ifdef CPPUDDLE_HAVE_COUNTERS
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
                << "--> Number of buffers that were marked as used upon "
                   "cleanup:      "
                << buffer_map.size() << std::endl
                << "==> Recycle rate:                                          "
                   "       "
                << static_cast<float>(number_recycling) / number_allocation *
                       100.0f
                << "%" << std::endl;
      // assert(buffer_map.size() == 0); // Were there any buffers still used?
#endif
      unused_buffer_list.clear();
      buffer_map.clear();
    }

  public: // Putting deleted constructors in public gives more useful error
          // messages
    // Bunch of constructors we don't need
    buffer_manager(
        buffer_manager<T, Host_Allocator> const &other) = delete;
    buffer_manager
    operator=(buffer_manager<T, Host_Allocator> const &other) = delete;
    buffer_manager(
        buffer_manager<T, Host_Allocator> &&other) = delete;
    buffer_manager
    operator=(buffer_manager<T, Host_Allocator> &&other) = delete;
  };

  /// Memory Manager subclass to handle buffers a specific type
  template <typename T, typename Host_Allocator>
  class mutexless_buffer_manager {
  private:
    // Tuple content: Pointer to buffer, buffer_size, reference_counter, Flag
    // The flag at the end controls whether to buffer content is to be reused as
    // well
    using buffer_entry_type = std::tuple<T *, size_t, size_t, bool>;

  public:
    /// Cleanup and delete this singleton
    static void clean() { manager_instance.reset(); }
    /// Cleanup all buffers not currently in use
    static void clean_unused_buffers_only() {
      if (!manager_instance) {
        return;
      }
      for (auto &buffer_tuple : manager_instance->unused_buffer_list) {
        Host_Allocator alloc;
        if (std::get<3>(buffer_tuple)) {
          util::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
      }
      manager_instance->unused_buffer_list.clear();
    }

    /// Tries to recycle or create a buffer of type T and size number_elements.
    static T *get(size_t number_of_elements, bool manage_content_lifetime) {
      if (!manager_instance) {
        manager_instance.reset(new mutexless_buffer_manager());
        buffer_recycler::add_total_cleanup_callback(clean);
        buffer_recycler::add_partial_cleanup_callback(
            clean_unused_buffers_only);
      }
#ifdef CPPUDDLE_HAVE_COUNTERS
      manager_instance->number_allocation++;
#endif
      // Check for unused buffers we can recycle:
      for (auto iter = manager_instance->unused_buffer_list.begin();
           iter != manager_instance->unused_buffer_list.end(); iter++) {
        auto tuple = *iter;
        if (std::get<1>(tuple) == number_of_elements) {
          manager_instance->unused_buffer_list.erase(iter);
          std::get<2>(tuple)++; // increase usage counter to 1

          // handle the switch from aggressive to non aggressive reusage (or
          // vice-versa)
          if (manage_content_lifetime && !std::get<3>(tuple)) {
            util::uninitialized_value_construct_n(std::get<0>(tuple),
                                                  number_of_elements);
            std::get<3>(tuple) = true;
          } else if (!manage_content_lifetime && std::get<3>(tuple)) {
            util::destroy_n(std::get<0>(tuple), std::get<1>(tuple));
            std::get<3>(tuple) = false;
          }
          manager_instance->buffer_map.insert({std::get<0>(tuple), tuple});
#ifdef CPPUDDLE_HAVE_COUNTERS
          manager_instance->number_recycling++;
#endif
          return std::get<0>(tuple);
        }
      }

      // No unsued buffer found -> Create new one and return it
      try {
        Host_Allocator alloc;
        T *buffer = alloc.allocate(number_of_elements);
        manager_instance->buffer_map.insert(
            {buffer, std::make_tuple(buffer, number_of_elements, 1,
                                     manage_content_lifetime)});
#ifdef CPPUDDLE_HAVE_COUNTERS
        manager_instance->number_creation++;
#endif
        if (manage_content_lifetime) {
          util::uninitialized_value_construct_n(buffer, number_of_elements);
        }
        return buffer;
      } catch (std::bad_alloc &e) {
        // not enough memory left! Cleanup and attempt again:
        buffer_recycler::clean_unused_buffers();

        // If there still isn't enough memory left, the caller has to handle it
        // We've done all we can in here
        Host_Allocator alloc;
        T *buffer = alloc.allocate(number_of_elements);
        manager_instance->buffer_map.insert(
            {buffer, std::make_tuple(buffer, number_of_elements, 1,
                                     manage_content_lifetime)});
#ifdef CPPUDDLE_HAVE_COUNTERS
        manager_instance->number_creation++;
        manager_instance->number_bad_alloc++;
#endif
        if (manage_content_lifetime) {
          util::uninitialized_value_construct_n(buffer, number_of_elements);
        }
        return buffer;
      }
    }

    static void mark_unused(T *memory_location, size_t number_of_elements) {
      // This will never be called without an instance since all access for this
      // method comes from the buffer recycler We can forego the instance
      // existence check here
      assert(manager_instance);
#ifdef CPPUDDLE_HAVE_COUNTERS
      manager_instance->number_dealloacation++;
#endif
      auto it = manager_instance->buffer_map.find(memory_location);
      assert(it != manager_instance->buffer_map.end());
      auto &tuple = it->second;
      // sanity checks:
      assert(std::get<1>(tuple) == number_of_elements);
      assert(std::get<2>(tuple) >= 1);
      std::get<2>(tuple)--;          // decrease usage counter
      if (std::get<2>(tuple) == 0) { // not used anymore?
        // move to the unused_buffer list
        manager_instance->unused_buffer_list.push_front(tuple);
        manager_instance->buffer_map.erase(memory_location);
      }
    }

    static void increase_usage_counter(T *memory_location,
                                       size_t number_of_elements) noexcept {
      auto it = manager_instance->buffer_map.find(memory_location);
      assert(it != manager_instance->buffer_map.end());
      auto &tuple = it->second;
      // sanity checks:
      assert(std::get<1>(tuple) == number_of_elements);
      assert(std::get<2>(tuple) >= 1);
      std::get<2>(tuple)++; // increase usage counter
    }

  private:
    /// List with all buffers still in usage
    std::unordered_map<T *, buffer_entry_type> buffer_map{};
    /// List with all buffers currently not used
    std::list<buffer_entry_type> unused_buffer_list{};
#ifdef CPPUDDLE_HAVE_COUNTERS
    /// Performance counters
    size_t number_allocation{0}, number_dealloacation{0};
    size_t number_recycling{0}, number_creation{0}, number_bad_alloc{0};
#endif
    /// Singleton instance
    static std::unique_ptr<mutexless_buffer_manager<T, Host_Allocator>>
        manager_instance;
    /// default, private constructor - not automatically constructed due to the
    /// deleted constructors
    mutexless_buffer_manager() = default;

  public:
    ~mutexless_buffer_manager() {
      for (auto &buffer_tuple : unused_buffer_list) {
        Host_Allocator alloc;
        if (std::get<3>(buffer_tuple)) {
          util::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
      }
      for (auto &map_tuple : buffer_map) {
        auto buffer_tuple = map_tuple.second;
        Host_Allocator alloc;
        if (std::get<3>(buffer_tuple)) {
          util::destroy_n(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
        }
        alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
      }
#ifdef CPPUDDLE_HAVE_COUNTERS
      // Print performance counters
      size_t number_cleaned = unused_buffer_list.size() + buffer_map.size();
      std::cout << "\nBuffer mananger destructor for buffers of type "
                << typeid(Host_Allocator).name() << "->" << typeid(T).name()
                << ":" << std::endl
                << "----------------------------------------------------"
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
                << "--> Number of buffers that were marked as used upon "
                   "cleanup:      "
                << buffer_map.size() << std::endl
                << "==> Recycle rate:                                          "
                   "       "
                << static_cast<float>(number_recycling) / number_allocation *
                       100.0f
                << "%" << std::endl;
      // assert(buffer_map.size() == 0); // Were there any buffers still used?
#endif
      unused_buffer_list.clear();
      buffer_map.clear();
    }

  public: // Putting deleted constructors in public gives more useful error
          // messages
    // Bunch of constructors we don't need
    mutexless_buffer_manager(
        mutexless_buffer_manager<T, Host_Allocator> const &other) = delete;
    mutexless_buffer_manager operator=(
        mutexless_buffer_manager<T, Host_Allocator> const &other) = delete;
    mutexless_buffer_manager(
        mutexless_buffer_manager<T, Host_Allocator> &&other) = delete;
    mutexless_buffer_manager
    operator=(mutexless_buffer_manager<T, Host_Allocator> &&other) = delete;
  };

public:
  // Putting deleted constructors in public gives more useful error messages
  // Bunch of constructors we don't need
  buffer_recycler(buffer_recycler const &other) = delete;
  buffer_recycler operator=(buffer_recycler const &other) = delete;
  buffer_recycler(buffer_recycler &&other) = delete;
  buffer_recycler operator=(buffer_recycler &&other) = delete;
};

template <typename T, typename Host_Allocator>
std::unique_ptr<buffer_recycler::buffer_manager<T, Host_Allocator>>
    buffer_recycler::buffer_manager<T, Host_Allocator>::manager_instance{};

template <typename T, typename Host_Allocator>
std::unique_ptr<buffer_recycler::mutexless_buffer_manager<T, Host_Allocator>>
    buffer_recycler::mutexless_buffer_manager<
        T, Host_Allocator>::manager_instance{};

template <typename T, typename Host_Allocator> struct recycle_allocator {
  using value_type = T;
  recycle_allocator() noexcept = default;
  template <typename U>
  explicit recycle_allocator(
      recycle_allocator<U, Host_Allocator> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data = buffer_recycler::get<T, Host_Allocator>(n);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n);
  }
  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    ::new (static_cast<void *>(p)) T(std::forward<Args>(args)...);
  }
  void destroy(T *p) { p->~T(); }
  void increase_usage_counter(T *p, size_t n) {
    buffer_recycler::increase_usage_counter<T, Host_Allocator>(p, n);
  }
};
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator==(recycle_allocator<T, Host_Allocator> const &,
           recycle_allocator<U, Host_Allocator> const &) noexcept {
  return true;
}
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator!=(recycle_allocator<T, Host_Allocator> const &,
           recycle_allocator<U, Host_Allocator> const &) noexcept {
  return false;
}

/// Recycles not only allocations but also the contents of a buffer
template <typename T, typename Host_Allocator>
struct aggressive_recycle_allocator {
  using value_type = T;
  aggressive_recycle_allocator() noexcept = default;
  template <typename U>
  explicit aggressive_recycle_allocator(
      aggressive_recycle_allocator<U, Host_Allocator> const &) noexcept {}
  T *allocate(std::size_t n) {
    T *data = buffer_recycler::get<T, Host_Allocator>(
        n, true); // also initializes the buffer if it isn't reused
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n);
  }
  template <typename... Args>
  inline void construct(T *p, Args... args) noexcept {
    // Do nothing here - we reuse the content of the last owner
  }
  void destroy(T *p) {
    // Do nothing here - Contents will be destroyed when the buffer manager is
    // destroyed, not before
  }
  void increase_usage_counter(T *p, size_t n) {
    buffer_recycler::increase_usage_counter<T, Host_Allocator>(p, n);
  }
};
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator==(aggressive_recycle_allocator<T, Host_Allocator> const &,
           aggressive_recycle_allocator<U, Host_Allocator> const &) noexcept {
  return true;
}
template <typename T, typename U, typename Host_Allocator>
constexpr bool
operator!=(aggressive_recycle_allocator<T, Host_Allocator> const &,
           aggressive_recycle_allocator<U, Host_Allocator> const &) noexcept {
  return false;
}

} // namespace detail

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_std = detail::recycle_allocator<T, std::allocator<T>>;
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_std =
    detail::aggressive_recycle_allocator<T, std::allocator<T>>;

/// Deletes all buffers (even ones still marked as used), delete the buffer
/// managers and the recycler itself
inline void force_cleanup() { detail::buffer_recycler::clean_all(); }
/// Deletes all buffers currently marked as unused
inline void cleanup() { detail::buffer_recycler::clean_unused_buffers(); }

} // end namespace recycler

#endif
