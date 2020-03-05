
#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <kokkos_viewpool.hpp>
#include <hpx/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

// scoped_timer -- stolen from Mikael
#include <hpx/timing/high_resolution_timer.hpp>
#include "../include/buffer_manager.hpp"
#include <memory>

//using kokkos_array = Kokkos::View<float[100], Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using kokkos_array = Kokkos::View<float[100], Kokkos::HostSpace>;
using kokkos_pinned_array = Kokkos::View<float[100], Kokkos::CudaHostPinnedSpace>;
using kokkos_cuda_array = Kokkos::View<float[100], Kokkos::CudaSpace>;

template <class kokkos_type, class alloc_type>
class recycled_view : public kokkos_type {
    private:
        static alloc_type allocator;
        size_t total_elements;
    public:
        template <class... Args>
        recycled_view(size_t total_elements, Args... args) : kokkos_type(allocator.allocate(total_elements),args...), total_elements(total_elements) {
        }
        ~recycled_view(void) {
            allocator.deallocate(this->data(), total_elements);
        }
};
template <class kokkos_type, class alloc_type>
alloc_type recycled_view<kokkos_type, alloc_type>::allocator;

// convience function to use the allocators together with Kokkos Views
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
    for (size_t i = 0; i < 100; i++) {
        (*input_array)(i) = i * 2.0;
    }

    // Way 1 to recycle heap buffer as well (manually)
    auto my_layout = input_array->layout();
    recycle_std<float> alli;
    float *my_recycled_data_buffer = alli.allocate(100); // allocate memory
    using kokkos_um_array = Kokkos::View<float[100], Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    {
        kokkos_um_array test_buffered(my_recycled_data_buffer);
        for (size_t i = 0; i < 100; i++) {
            test_buffered(i) = i * 2.0;
        }
    }
    alli.deallocate(my_recycled_data_buffer, 100); 
    size_t to_alloc = kokkos_um_array::required_allocation_size(100);
    std::cout << "Actual required size: "  << to_alloc << std::endl; // Still a heap allocation!

    // Way 2 to recycle 
    recycled_view<kokkos_um_array, recycle_std<float>> my_wrapper_test0(100);
    for (size_t i = 0; i < 100; i++) {
        my_wrapper_test0(i) = i * 2.0;
    }

    { // Just some views that will be destroyed again to be recylced in the next block
        recycled_view<kokkos_um_array, recycle_std<float>> my_wrapper_test1(100);
        recycled_view<kokkos_um_array, recycle_std<float>> my_wrapper_test2(100);
    }
    { // Let the recycling commence
        recycled_view<kokkos_um_array, recycle_std<float>> my_wrapper_test1(100);
        recycled_view<kokkos_um_array, recycle_std<float>> my_wrapper_test2(100);
        recycled_view<kokkos_um_array, recycle_std<float>> my_wrapper_test3(100);
    }
}