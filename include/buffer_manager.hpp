// TODO Insert Allocator

// TODO Insert singleton memory mananger -> Wird der überhaupt benötigt?
// Vermutlich hauptsächlich um callbacks zu verwalten und die Synchronisierung zu machen
#include <iostream>


class buffer_recycler {
  // public interface
  public:
    template <class T>
    static void get(size_t number_elements) {
      if (!instance)
        instance = new buffer_recycler();
        buffer_manager<T>::get(number_elements);
    }
    static void clean(void) {
      if (instance) {
        for (auto clean_function : instance->total_cleanup_callbacks)
          clean_function();
        delete instance;
      }
    }

    // TODO I'd like to only be able to call these from the buffer_managers, not from anywhere else
    static void add_total_cleanup_callback(std::function<void()> func) {
      if (instance)
        instance->total_cleanup_callbacks.push_back(func);
    }
    // TODO I'd like to only be able to call these from the buffer_managers, not from anywhere else
    static void add_partial_cleanup_callback(std::function<void()> func) {
      if (instance)
        instance->partial_cleanup_callbacks.push_back(func);
    }

    // Bunch of constructors we don't need
    buffer_recycler(buffer_recycler &other) = delete;
    buffer_recycler(buffer_recycler const &other) = delete;
    buffer_recycler operator=(buffer_recycler const &other) = delete;
    buffer_recycler operator=(buffer_recycler &other) = delete;
    buffer_recycler(buffer_recycler &&other) = delete;
    buffer_recycler(buffer_recycler const &&other) = delete;
    buffer_recycler operator=(buffer_recycler const &&other) = delete;
    buffer_recycler operator=(buffer_recycler &&other) = delete;

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
      std::cout << "Buffer recycler destructor!" << std::endl;
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
        }
        static void clean_unused_buffers_only(void) {
          if (!instance)
            return;
          // Check buffers in a for loop and delete unused ones
        }

        /// Tries to recycle or create a buffer of type T and size number_elements. 
        static void get(size_t number_elements) {
          if (!instance) {
            instance = new buffer_manager();
            buffer_recycler::add_total_cleanup_callback(clean);
            buffer_recycler::add_partial_cleanup_callback(clean_unused_buffers_only);
          }
          // TODO Check for unused buffers we can recycle:

          // No unsued buffer found -> Create new one
          instance->buffer_list.push_back(new T[number_elements]);

          // TODO Return new buffer
        }

        // Bunch of constructors we don't need
        buffer_manager<T>(buffer_manager<T> &other) = delete;
        buffer_manager<T>(buffer_manager<T> const &other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> const &other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> &other) = delete;
        buffer_manager<T>(buffer_manager<T> &&other) = delete;
        buffer_manager<T>(buffer_manager<T> const &&other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> const &&other) = delete;
        buffer_manager<T> operator=(buffer_manager<T> &&other) = delete;

      // TODO Delete other constructors
      private:
        std::list<T*> buffer_list;
        static buffer_manager<T> *instance; 

        buffer_manager(void) {
          std::cout << "Buffer mananger constructor for " << typeid(T).name() << std::endl;
        }
        ~buffer_manager(void) {
          std::cout << "Buffer mananger destructor for " << typeid(T).name() << std::endl;
          for (T *buffer : buffer_list) {
            delete [] buffer;
          }
          buffer_list.clear();
        }
    };
};

// Instance defintions
buffer_recycler* buffer_recycler::instance = nullptr;

template<class T>
buffer_recycler::buffer_manager<T>* buffer_recycler::buffer_manager<T>::instance = nullptr; 