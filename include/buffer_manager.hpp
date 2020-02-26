#include <iostream>
#include <mutex>

class buffer_recycler {
  // Public interface
  public:
    template <class T>
    static T* get(size_t number_elements) {
      std::lock_guard<std::mutex> guard(mut);
      if (!instance) {
        instance = new buffer_recycler();
        destroyer.set_singleton(instance);
      }
      return buffer_manager<T>::get(number_elements);
    }

    template <class T>
    static void mark_unused(T *p, size_t number_elements) {
      std::lock_guard<std::mutex> guard(mut);
      if (!instance) {
        instance = new buffer_recycler();
        destroyer.set_singleton(instance);
      }
      return buffer_manager<T>::mark_unused(p,number_elements);
    }
    static void clean_all(void) {
      std::lock_guard<std::mutex> guard(mut);
      if (instance) {
        delete instance;
        instance = nullptr;
        destroyer.set_singleton(nullptr);
      }
    }
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

    buffer_recycler(void) {
      std::cout << "Buffer recycler constructor!" << std::endl;
    }
    ~buffer_recycler(void) {
      for (auto clean_function : total_cleanup_callbacks)
        clean_function();
      std::cout << "Buffer recycler destructor!" << std::endl;
    }

    static void add_total_cleanup_callback(std::function<void()> func) {
        // This methods assumes instance is initialized since it is a private method and all static public methods have guards
        instance->total_cleanup_callbacks.push_back(func);
    }

    static void add_partial_cleanup_callback(std::function<void()> func) {
        // This methods assumes instance is initialized since it is a private method and all static public methods have guards
        instance->partial_cleanup_callbacks.push_back(func);
    }


  // Subclasses
  private: 
    /// Memory Manager subclass to handle buffers of specific types and sizes
    template<class T>
    class buffer_manager {
      public:
        static void clean(void) {
          if (!instance)
            return;
          delete instance;
          instance = nullptr;
        }
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
          // Check for unused buffers we can recycle:
          for (auto iter = instance->unused_buffer_list.begin(); iter != instance->unused_buffer_list.end(); iter++) {
            if (std::get<1>(*iter) == number_of_elements) {
              instance->buffer_list.push_back(*iter);
              instance->unused_buffer_list.erase(iter);
              std::cout << "-->Recycled buffer" << std::endl;
              return std::get<0>(instance->buffer_list.back());
            }
          }

          // No unsued buffer found -> Create new one and return it
          try {
            T *buffer = new T[number_of_elements];
            instance->buffer_list.push_back(std::make_tuple(buffer, number_of_elements));
            std::cout << "-->Created new buffer" << std::endl;
            return std::get<0>(instance->buffer_list.back());
          }
          catch(std::bad_alloc &e) { 
            // not enough memory left! Cleanup and attempt again:
            buffer_recycler::clean_unused_buffers();

            // If there still isn't enough memory left, the caller has to handle it 
            // We've done all we can in here
            T *buffer = new T[number_of_elements];
            instance->buffer_list.push_back(std::make_tuple(buffer, number_of_elements));
            std::cout << "-->Created new buffer after bad_alloc" << std::endl;
            return std::get<0>(instance->buffer_list.back());
          }
        }

        static void mark_unused(T* memory_location, size_t number_of_elements) {
          // Search for used buffer
          auto to_mark = std::make_tuple(memory_location, number_of_elements);
          for (auto iter = instance->buffer_list.begin(); iter != instance->buffer_list.end(); iter++) {
            if (*iter == to_mark) {
              instance->unused_buffer_list.push_front(to_mark);
              instance->buffer_list.erase(iter);
              return;
            }
          }
          const char *error_message =R""""(
            Error! Deallocate was called on a memory location that is not known to the buffer_manager!\n
            This should never happen!
          )""""; 
          throw std::logic_error(error_message);
        }

      private:
        std::list<std::tuple<T*,size_t>> buffer_list; // used buffers
        std::list<std::tuple<T*,size_t>> unused_buffer_list; // unused buffers
        static buffer_manager<T> *instance; 

        buffer_manager(void) {
          std::cout << "Buffer mananger constructor for buffers of type " << typeid(T).name() << "!" << std::endl;
        }
        ~buffer_manager(void) {
          for (auto buffer_tuple : unused_buffer_list) {
            delete [] std::get<0>(buffer_tuple);
          }
          for (auto buffer_tuple : buffer_list) {
            delete [] std::get<0>(buffer_tuple);
          }
          if (buffer_list.size() > 0) {
            const char *error_message =R""""(
              WARNING: Some buffers are still marked as used upon the destruction of the buffer_manager!
              Please check if you are using the buffer_recycler without the recycle_allocator.
              If yes, you can probably fix this by manually marking buffers as unused!
            )""""; 
            //throw std::logic_error(error_message);
            std::cerr << error_message << std::endl;
          }

          std::cout << "Buffer mananger destructor for buffers of type " << typeid(T).name() 
                    << "!\n-->Deleted " << unused_buffer_list.size()  << " unused buffers! " << std::endl
                    << "-->Deleted " << buffer_list.size()  << " still used buffers! " << std::endl;
          unused_buffer_list.clear();
          buffer_list.clear();
        }

      public: // Putting deleted constructors in public gives more useful error messages
        // Bunch of constructors we don't need
        buffer_manager<T>(buffer_manager<T> const &other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> const &other) = delete;
        buffer_manager<T>(buffer_manager<T> const &&other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> const &&other) = delete;
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
    buffer_recycler(buffer_recycler const &&other) = delete;
    buffer_recycler operator=(buffer_recycler const &&other) = delete;
};

// Instance defintions
buffer_recycler* buffer_recycler::instance = nullptr;
buffer_recycler::memory_manager_destroyer buffer_recycler::destroyer;
std::mutex buffer_recycler::mut;

template<class T>
buffer_recycler::buffer_manager<T>* buffer_recycler::buffer_manager<T>::instance = nullptr; 

template <class T>
struct recycle_allocator {
  using value_type = T;
  recycle_allocator() noexcept {}
  template <class U>
  recycle_allocator(recycle_allocator<U> const&) noexcept {
  }
  T* allocate(std::size_t n) {
    std::cout << "calling allocate" << std::endl;
    T* data = buffer_recycler::get<T>(n);
    return data;
  }
  void deallocate(T *p, std::size_t n) {
    std::cout << "calling deallocate" << std::endl;
    buffer_recycler::mark_unused<T>(p, n);
  }
};

template <class T, class U>
constexpr bool operator==(recycle_allocator<T> const&, recycle_allocator<U> const&) noexcept {
  return true;
}
template <class T, class U>
constexpr bool operator!=(recycle_allocator<T> const&, recycle_allocator<U> const&) noexcept {
  return false;
}