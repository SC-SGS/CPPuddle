#include <iostream>

class buffer_recycler {
  // Public interface
  public:
    template <class T>
    static void get(size_t number_elements) {
      if (!instance) {
        instance = new buffer_recycler();
        destroyer.set_singleton(instance);
      }
      buffer_manager<T>::get(number_elements);
    }
    static void clean_all(void) {
      if (instance) {
        delete instance;
        instance = nullptr;
        destroyer.set_singleton(nullptr);
      }
    }
    static void clean_unused_buffers(void) {
      if (instance) {
        for (auto clean_function : instance->total_cleanup_callbacks)
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
          // Check buffers in a for loop and delete unused ones
        }

        /// Tries to recycle or create a buffer of type T and size number_elements. 
        static void get(size_t number_of_elements) {
          if (!instance) {
            instance = new buffer_manager();
            buffer_recycler::add_total_cleanup_callback(clean);
            buffer_recycler::add_partial_cleanup_callback(clean_unused_buffers_only);
          }
          // TODO Check for unused buffers we can recycle:

          // No unsued buffer found -> Create new one
          instance->buffer_list.push_back(new T[number_of_elements]);

          // TODO Return new buffer
        }

      private:
        std::list<T*> buffer_list;
        static buffer_manager<T> *instance; 

        buffer_manager(void) {
          std::cout << "Buffer mananger constructor for buffers of type " << typeid(T).name() << "!" << std::endl;
        }
        ~buffer_manager(void) {
          for (T *buffer : buffer_list) {
            delete [] buffer;
          }
          std::cout << "Buffer mananger destructor for buffers of type " << typeid(T).name() 
                    << "! Deleted " << buffer_list.size()  << " buffers! " << std::endl;
          buffer_list.clear();
        }

      public: // Putting deleted constructors in public gives more useful error messages
        // Bunch of constructors we don't need
        buffer_manager<T>(buffer_manager<T> &other) = delete;
        buffer_manager<T>(buffer_manager<T> const &other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> const &other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> &other) = delete;
        buffer_manager<T>(buffer_manager<T> &&other) = delete;
        buffer_manager<T>(buffer_manager<T> const &&other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> const &&other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> &&other) = delete;
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
    buffer_recycler(buffer_recycler &other) = delete;
    buffer_recycler(buffer_recycler const &other) = delete;
    buffer_recycler operator=(buffer_recycler const &other) = delete;
    buffer_recycler operator=(buffer_recycler &other) = delete;
    buffer_recycler(buffer_recycler &&other) = delete;
    buffer_recycler(buffer_recycler const &&other) = delete;
    buffer_recycler operator=(buffer_recycler const &&other) = delete;
    buffer_recycler operator=(buffer_recycler &&other) = delete;

};

// Instance defintions
buffer_recycler* buffer_recycler::instance = nullptr;
buffer_recycler::memory_manager_destroyer buffer_recycler::destroyer;

template<class T>
buffer_recycler::buffer_manager<T>* buffer_recycler::buffer_manager<T>::instance = nullptr; 