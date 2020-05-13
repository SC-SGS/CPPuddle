#pragma once

#include <iostream>
#include <mutex>
#include <memory>

class buffer_recycler {
  // Public interface
  public:
    /// Returns and allocated buffer of the requested size - this may be a reused buffer
    template <typename T, typename Host_Allocator>
    static T* get(size_t number_elements) {
      std::lock_guard<std::mutex> guard(mut);
      if (!instance) {
        instance = new buffer_recycler();
        destroyer.set_singleton(instance);
      }
      return buffer_manager<T, Host_Allocator>::get(number_elements);
    }
    /// Marks an buffer as unused and fit for reusage
    template <typename T, typename Host_Allocator>
    static void mark_unused(T *p, size_t number_elements) {
      std::lock_guard<std::mutex> guard(mut);
      if (!instance) {
        instance = new buffer_recycler();
        destroyer.set_singleton(instance);
      }
      return buffer_manager<T, Host_Allocator>::mark_unused(p,number_elements);
    }
    /// Increase the reference coutner of a buffer
    template <typename T, typename Host_Allocator>
    static void increase_usage_counter(T *p, size_t number_elements) {
      std::lock_guard<std::mutex> guard(mut);
      assert(instance != nullptr);
      return buffer_manager<T, Host_Allocator>::increase_usage_counter(p,number_elements);
    }
    /// Deallocated all buffers, no matter whether they are marked as used or not
    static void clean_all(void) {
      std::lock_guard<std::mutex> guard(mut);
      if (instance) {
        delete instance;
        instance = nullptr;
        destroyer.set_singleton(nullptr);
      }
    }
    /// Deallocated all currently unused buffer
    static void clean_unused_buffers(void) {
      std::lock_guard<std::mutex> guard(mut);
      if (instance) {
        for (auto clean_function : instance->partial_cleanup_callbacks)
          clean_function();
      }
    }

  // Member variables and methods
  private: 
    /// Singleton instance pointer
    static buffer_recycler *instance;
    /// Callbacks for buffer_manager cleanups - each callback complete destroys one buffer_manager
    std::list<std::function<void()>> total_cleanup_callbacks;
    /// Callbacks for partial buffer_manager cleanups - each callback deallocates all unsued buffers of a manager
    std::list<std::function<void()>> partial_cleanup_callbacks;
    /// One Mutex to control concurrent access - Since we do not actually ever return the singleton instance anywhere, this should hopefully suffice
    /// We want more fine-grained concurrent access eventually
    static std::mutex mut;
    /// default, private constructor - not automatically constructed due to the deleted constructors
    buffer_recycler(void) = default;
    /// Clean all buffers by using the callbacks of the buffer managers
    ~buffer_recycler(void) {
      for (auto clean_function : total_cleanup_callbacks)
        clean_function();
    }
    /// Add a callback function that gets executed upon cleanup and destruction
    static void add_total_cleanup_callback(std::function<void()> func) {
        // This methods assumes instance is initialized since it is a private method and all static public methods have guards
        instance->total_cleanup_callbacks.push_back(func);
    }
    /// Add a callback function that gets executed upon partial (unused memory) cleanup
    static void add_partial_cleanup_callback(std::function<void()> func) {
        // This methods assumes instance is initialized since it is a private method and all static public methods have guards
        instance->partial_cleanup_callbacks.push_back(func);
    }

  // Subclasses
  private: 
    /// Memory Manager subclass to handle buffers a specific type 
    template<typename T, typename Host_Allocator>
    class buffer_manager {
      public:
        /// Cleanup and delete this singleton
        static void clean(void) {
          if (!instance)
            return;
          delete instance;
          instance = nullptr;
        }
        /// Cleanup all buffers not currently in use
        static void clean_unused_buffers_only(void) {
          if (!instance)
            return;
          for (auto buffer_tuple : instance->unused_buffer_list) {
            delete [] std::get<0>(buffer_tuple);
          }
          instance->unused_buffer_list.clear();
        }

        /// Tries to recycle or create a buffer of type T and size number_elements. 
        static T* get(size_t number_of_elements) {
          if (!instance) {
            instance = new buffer_manager();
            buffer_recycler::add_total_cleanup_callback(clean);
            buffer_recycler::add_partial_cleanup_callback(clean_unused_buffers_only);
          }
          instance->number_allocation++;
          // Check for unused buffers we can recycle:
          for (auto iter = instance->unused_buffer_list.begin(); iter != instance->unused_buffer_list.end(); iter++) {
            auto tuple = *iter;
            if (std::get<1>(tuple) == number_of_elements) {
              instance->unused_buffer_list.erase(iter);
              std::get<2>(tuple)++; // increase usage counter to 1
              instance->buffer_map.insert({std::get<0>(tuple), tuple});
              instance->number_recycling++;
              return std::get<0>(tuple);
            }
          }

          // No unsued buffer found -> Create new one and return it
          try {
            //T *buffer = new T[number_of_elements];
            Host_Allocator alloc;
            T *buffer = alloc.allocate(number_of_elements);
            instance->buffer_map.insert({buffer, std::make_tuple(buffer, number_of_elements, 1)});
            instance->number_creation++;
            return buffer;
          }
          catch(std::bad_alloc &e) { 
            // not enough memory left! Cleanup and attempt again:
            buffer_recycler::clean_unused_buffers();

            // If there still isn't enough memory left, the caller has to handle it 
            // We've done all we can in here
            //T *buffer = new T[number_of_elements];
            Host_Allocator alloc;
            T *buffer = alloc.allocate(number_of_elements);
            instance->buffer_map.insert({buffer, std::make_tuple(buffer, number_of_elements, 1)});
            instance->number_creation++;
            instance->number_bad_alloc++;
            return buffer;
          }
        }

        static void mark_unused(T* memory_location, size_t number_of_elements) {
          // This will never be called without an instance since all access for this method comes from the buffer recycler 
          // We can forego the instance existence check here
          instance->number_dealloacation++;
          auto it = instance->buffer_map.find(memory_location);
          assert(it != instance->buffer_map.end());
          auto &tuple = it->second;
          // sanity checks:
          assert(std::get<1>(tuple) == number_of_elements);
          assert(std::get<2>(tuple) >= 1);
          std::get<2>(tuple)--; // decrease usage counter
          if (std::get<2>(tuple) == 0) { // not used anymore?
            // move to the unused_buffer list 
            instance->unused_buffer_list.push_front(tuple);
            instance->buffer_map.erase(memory_location);
          }
        }

        static void increase_usage_counter(T* memory_location, size_t number_of_elements) {
          auto it = instance->buffer_map.find(memory_location);
          assert(it != instance->buffer_map.end());
          auto &tuple = it->second;
          // sanity checks:
          assert(std::get<1>(tuple) == number_of_elements);
          assert(std::get<2>(tuple) >= 1);
          std::get<2>(tuple)++; // increase usage counter
        }

      private:
        /// List with all buffers still in usage 
        std::unordered_map<T*, std::tuple<T*, size_t, size_t>> buffer_map;
        /// List with all buffers currently not used
        std::list<std::tuple<T*,size_t, size_t>> unused_buffer_list; 
        /// Performance counters
        size_t number_allocation{0}, number_dealloacation{0};
        size_t number_recycling{0}, number_creation{0}, number_bad_alloc{0};
        /// Singleton instance
        static buffer_manager<T, Host_Allocator> *instance; 
        /// default, private constructor - not automatically constructed due to the deleted constructors
        buffer_manager(void) = default;
        ~buffer_manager(void) {
          for (auto &buffer_tuple : unused_buffer_list) {
            Host_Allocator alloc;
            alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
          }
          for (auto &map_tuple : buffer_map) {
            auto buffer_tuple = map_tuple.second;
            Host_Allocator alloc;
            alloc.deallocate(std::get<0>(buffer_tuple), std::get<1>(buffer_tuple));
          }
          // Print performance counters
          size_t number_cleaned = unused_buffer_list.size() + buffer_map.size();
          std::cout << "\nBuffer mananger destructor for buffers of type " << typeid(T).name() << ":" << std::endl
                    << "----------------------------------------------------" << std::endl
                    << "--> Number of bad_allocs that triggered garbage collection:       " << number_bad_alloc << std::endl
                    << "--> Number of buffers that got requested from this manager:       " << number_allocation << std::endl
                    << "--> Number of times an unused buffer got recycled for a request:  " << number_recycling << std::endl
                    << "--> Number of times a new buffer had to be created for a request: " << number_creation << std::endl 
                    << "--> Number cleaned up buffers:                                    " << number_cleaned << std::endl 
                    << "--> Number of buffers that were marked as used upon cleanup:      " << buffer_map.size() << std::endl
                    << "==> Recycle rate:                                                 " 
                    << static_cast<float>(number_recycling)/number_allocation * 100.0f << "%" << std::endl;
          assert(buffer_map.size() == 0); // Were there any buffers still used? 
          unused_buffer_list.clear();
          buffer_map.clear();
        }

      public: // Putting deleted constructors in public gives more useful error messages
        // Bunch of constructors we don't need
        buffer_manager<T, Host_Allocator>(buffer_manager<T, Host_Allocator> const &other) = delete;
        buffer_manager<T, Host_Allocator> operator=(buffer_manager<T, Host_Allocator> const &other) = delete;
        buffer_manager<T, Host_Allocator>(buffer_manager<T, Host_Allocator> &&other) = delete;
        buffer_manager<T, Host_Allocator> operator=(buffer_manager<T, Host_Allocator> &&other) = delete;
    };

    /// This class just makes sure the singleton is destroyed automatically UNLESS it has already been explictly destroyed
    /** A user might want to explictly destroy all buffers, for example before a Kokkos cleanup.
     * However, we also want to clean up all buffers when the static variables of the program are destroyed. 
     * Having a static instance of this in the buffer_recycler ensures the latter part whilst still maintaining
     * the possibiltiy for manual cleanup using buffer_recycler::clean_all
     */
    class memory_manager_destroyer {
      public:
        memory_manager_destroyer(buffer_recycler *instance = nullptr) {
          singleton = instance;
        }
        ~memory_manager_destroyer() {
          if (singleton != nullptr)
            delete singleton;
          singleton = nullptr;
        }
        void set_singleton(buffer_recycler *s) {
          singleton = s;
        }
      private:
        buffer_recycler *singleton;
    };
    /// Static instance of the destroyer - gets destroyed at the end of the program and kills any remaining buffer_recycler with it
    static memory_manager_destroyer destroyer;

  public: // Putting deleted constructors in public gives more useful error messages
    // Bunch of constructors we don't need
    buffer_recycler(buffer_recycler const &other) = delete;
    buffer_recycler operator=(buffer_recycler const &other) = delete;
    buffer_recycler(buffer_recycler &&other) = delete;
    buffer_recycler operator=(buffer_recycler &&other) = delete;
};

// Instance defintions
buffer_recycler* buffer_recycler::instance = nullptr;
buffer_recycler::memory_manager_destroyer buffer_recycler::destroyer;
std::mutex buffer_recycler::mut;

template<typename T, typename Host_Allocator>
buffer_recycler::buffer_manager<T, Host_Allocator>* buffer_recycler::buffer_manager<T, Host_Allocator>::instance = nullptr; 

template <typename T, typename Host_Allocator>
struct recycle_allocator {
  using value_type = T;
  recycle_allocator() noexcept {}
  template <typename U>
  recycle_allocator(recycle_allocator<U, Host_Allocator> const&) noexcept {
  }
  T* allocate(std::size_t n) {
    T* data = buffer_recycler::get<T, Host_Allocator>(n);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    buffer_recycler::mark_unused<T, Host_Allocator>(p, n);
  }
  template<typename... Args>
  void construct(T *p, Args... args) {
    ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
  }
  void destroy(T *p) {
    p->~T();
  }
  void increase_usage_counter(T *p, size_t n) {
    buffer_recycler::increase_usage_counter<T, Host_Allocator>(p, n);
  }
};

template <typename T, typename U, typename Host_Allocator>
constexpr bool operator==(recycle_allocator<T, Host_Allocator> const&, recycle_allocator<U, Host_Allocator> const&) noexcept {
  return true;
}
template <typename T, typename U, typename Host_Allocator>
constexpr bool operator!=(recycle_allocator<T, Host_Allocator> const&, recycle_allocator<U, Host_Allocator> const&) noexcept {
  return false;
}

template<typename T>
using recycle_std = recycle_allocator<T, std::allocator<T>>;
