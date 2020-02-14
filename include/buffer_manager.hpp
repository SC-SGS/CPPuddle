// TODO Insert Allocator

// TODO Insert singleton memory mananger -> Wird der überhaupt benötigt?
// Vermutlich hauptsächlich um callbacks zu verwalten und die Synchronisierung zu machen
#include <iostream>


class buffer_recycler {
  public:
    template <class T>
    static void get(size_t number_elements) {
      if (!instance)
        instance = new buffer_recycler();
        buffer_manager<T>::get(number_elements);
    }
    static void clean(void) {
      if (instance)
        delete instance;
    }

  // Bunch of constructors we don't need
  public:
    buffer_recycler(buffer_recycler &other) = delete;
    buffer_recycler(buffer_recycler const &other) = delete;
    buffer_recycler operator=(buffer_recycler const &other) = delete;
    buffer_recycler operator=(buffer_recycler &other) = delete;
    buffer_recycler(buffer_recycler &&other) = delete;
    buffer_recycler(buffer_recycler const &&other) = delete;
    buffer_recycler operator=(buffer_recycler const &&other) = delete;
    buffer_recycler operator=(buffer_recycler &&other) = delete;

  private: 
    /// Singleton instance pointer
    static buffer_recycler *instance;
    /// Callbacks for buffer_manager cleanups
    std::list<std::cuntion
    buffer_recycler(void) {
      std::cout << "Buffer recycler constructor!" << std::endl;
    }
    ~buffer_recycler(void) {
      std::cout << "Buffer recycler destructor!" << std::endl;
    }



  private: 
    /// Memory Manager subclass to handle buffers of specific types and sizes
    template<class T>
    class buffer_manager {
      public:
        /// Tries to recycle or create a buffer of type T and size number_elements. 
        static void get(size_t number_elements) {
          if (!instance)
            instance = new buffer_manager();
          // TODO Check for unused buffers:

          // No unsued buffer found -> Create new one
          instance->buffer_list.push_back(new T[number_elements]);

          // TODO Return new buffer
        }

      // Bunch of constructors we don't need
      public:
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
        }
    };
};

// Instance defintions
buffer_recycler* buffer_recycler::instance = nullptr;

template<class T>
buffer_recycler::buffer_manager<T>* buffer_recycler::buffer_manager<T>::instance = nullptr; 