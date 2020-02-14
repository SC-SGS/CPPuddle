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
  private: 
    static buffer_recycler *instance;
    buffer_recycler(void) {
      std::cout << "Buffer recycler constructor!" << std::endl;
    }
    ~buffer_recycler(void) {
      std::cout << "Buffer recycler destructor!" << std::endl;
    }

  private: 
    template<class T>
    class buffer_manager {
      public:
        static void get(size_t number_elements) {
        if (!instance)
            instance = new buffer_manager();
        }

      private:
        std::list<T*> buffer_list;
        static buffer_manager<T> *instance; // requires C++17

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