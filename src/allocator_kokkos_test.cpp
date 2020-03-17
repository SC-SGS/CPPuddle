
#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <hpx/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

// scoped_timer -- stolen from Mikael
#include "../include/buffer_manager.hpp"
#include <hpx/timing/high_resolution_timer.hpp>
#include <memory>

//using kokkos_array = Kokkos::View<float[1000], Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
constexpr size_t view_size_0 = 100;
constexpr size_t view_size_1 = 1000;
using type_in_view = float[view_size_1][view_size_0];
constexpr size_t view_size = view_size_0*view_size_1;
using kokkos_array = Kokkos::View<type_in_view, Kokkos::HostSpace>;
// using kokkos_pinned_array = Kokkos::View<type_in_view, Kokkos::CudaHostPinnedSpace>;
// using kokkos_cuda_array = Kokkos::View<type_in_view, Kokkos::CudaSpace>;

template <class kokkos_type, class alloc_type, class element_type>
class recycled_view : public kokkos_type {
    private:
        static alloc_type allocator;
        size_t total_elements;
    public:
        template <class... Args>
        recycled_view(Args... args) :
          kokkos_type(allocator.allocate(kokkos_type::required_allocation_size(args...) / sizeof(element_type)),args...),
          total_elements(kokkos_type::required_allocation_size(args...) / sizeof(element_type)) {
            //std::cout << "Got buffer for " << total_elements << std::endl;
        }
        recycled_view(const recycled_view<kokkos_type, alloc_type, element_type> &other) : 
          kokkos_type(other) {
          allocator.increase_usage_count(other.data(), other.total_elements);
        }
        recycled_view<kokkos_type, alloc_type, element_type>& operator = (const recycled_view<kokkos_type, alloc_type, element_type> &other) {
          kokkos_type::operator = (other);
          allocator.increase_usage_count(other.data(), other.total_elements);
          return *this;
        }
        recycled_view(recycled_view<kokkos_type, alloc_type, element_type> &&other) : 
          kokkos_type(other) {
          // so that is doesn't matter if deallocate is called in the moved-from object
          allocator.increase_usage_count(other.data(), other.total_elements);
        }
        recycled_view<kokkos_type, alloc_type, element_type>& operator = (recycled_view<kokkos_type, alloc_type, element_type> &&other) {
          kokkos_type::operator = (other);
          // so that is doesn't matter if deallocate is called in the moved-from object
          allocator.increase_usage_count(other.data(), other.total_elements);
          return *this;
        }
        ~recycled_view(void) {
            allocator.deallocate(this->data(), total_elements);
        }
};
template <class kokkos_type, class alloc_type, class element_type>
alloc_type recycled_view<kokkos_type, alloc_type, element_type>::allocator;

// Just some 2D views used for testing
template <class T>
using kokkos_um_array = Kokkos::View<T**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view = recycled_view<kokkos_um_array<T>, recycle_std<T>, T>;

template <class T>
using kokkos_um_device_array = Kokkos::View<T**, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_device_view = recycled_view<kokkos_um_device_array<T>, recycle_allocator_cuda_device<T>, T>;


// convenience function to use the allocators together with Kokkos Views
template <class T, class... Args>
std::unique_ptr<T, std::function<void(T *)>> make_or_recycle_unique(Args... args)
{
    auto deleter = [](T *p) {
        recycle_std<T> alloc;
        alloc.destroy(p);
        alloc.deallocate(p, 1);
    };
    recycle_std<T> alloc;
    T *ptr = alloc.allocate(1);
    alloc.construct(ptr, std::forward<Args>(args)...);
    return std::unique_ptr<T, std::function<void(T *)>>(ptr, deleter);
}

// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{
    hpx::kokkos::ScopeGuard scopeGuard(argc, argv);
    Kokkos::print_configuration(std::cout);

    std::cout << "Size: "  << sizeof(kokkos_array) << std::endl; // Still a heap allocation!

    // Bad way: Does not recycle the heap buffer of Kokkos as of yet
    auto input_array = make_or_recycle_unique<kokkos_array>(std::string("my_smart_view"));
    for (size_t i = 0; i < view_size; i++) {
        (*input_array).data()[i] = i * 2.0;
    }

    // Way 1 to recycle heap buffer as well (manually)
    auto my_layout = input_array->layout();
    recycle_std<float> alli;
    float *my_recycled_data_buffer = alli.allocate(view_size); // allocate memory
    {
        kokkos_um_array<float> test_buffered(my_recycled_data_buffer, view_size_0, view_size_1);
        for (size_t i = 0; i < view_size; i++) {
            test_buffered.data()[i] = i * 2.0;
        }
    }
    alli.deallocate(my_recycled_data_buffer, view_size); 
    size_t to_alloc = kokkos_um_array<float>::required_allocation_size(view_size);
    std::cout << "Actual required size: "  << to_alloc << std::endl; // Still a heap allocation!

    // Way 2 for recycling 
    using test_view = recycled_host_view<float>;
    using test_double_view = recycled_host_view<double>;
    test_view my_wrapper_test0(view_size_0, view_size_1);
    for (size_t i = 0; i < view_size; i++) {
        my_wrapper_test0.data()[i] = i * 2.0;
    }

    // for some views on cuda data
    using test_device_view = recycled_device_view<float>;
    using test_device_double_view = recycled_device_view<double>;


    /** Stress test for safe concurrency and performance:
   *  stolen from allocator_test
   * */

    constexpr size_t number_futures = 64;
    constexpr size_t passes = 10;

    static_assert(passes >= 0);
    static_assert(view_size >= 1);
    assert(number_futures >= hpx::get_num_worker_threads());

    auto begin = std::chrono::high_resolution_clock::now();
    std::array<hpx::future<void>, number_futures> futs;
    for (size_t i = 0; i < number_futures; i++)
    {
        futs[i] = hpx::make_ready_future<void>();
    }
    for (size_t pass = 0; pass < passes; pass++)
    {
        for (size_t i = 0; i < number_futures; i++)
        {
            futs[i] = futs[i].then([&](hpx::future<void> &&predecessor) {
                // now we can even swap the sizes at runtime, will still be reused
                test_view test0(view_size_0, view_size_1);
                test_view test1(view_size_1, view_size_0);
                test_double_view test2(view_size_0, view_size_1);
                test_double_view test3(view_size_1, view_size_0);
                test_device_view test4(view_size_0, view_size_1);
                test_device_view test5(view_size_1, view_size_0);
                test_device_double_view test6(view_size_0, view_size_1);
                test_device_double_view test7(view_size_1, view_size_0);
            });
        }
    }
    auto when = hpx::when_all(futs);
    when.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\n==>Allocation test took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
}
