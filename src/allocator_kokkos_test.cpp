
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

using kokkos_array = Kokkos::View<float[100], Kokkos::HostSpace>;
using kokkos_pinned_array = Kokkos::View<float[100], Kokkos::CudaHostPinnedSpace>;
using kokkos_cuda_array = Kokkos::View<float[100], Kokkos::CudaSpace>;

// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{
    hpx::kokkos::ScopeGuard scopeGuard(argc, argv);
    Kokkos::print_configuration(std::cout);


    // Current Option 1: (the smart (pointer) road)
    auto input_array = make_or_recycle_unique<kokkos_array>(std::string("my_smart_view"));
    for (size_t i = 0; i < 100; i++) {
        (*input_array)(i) = i * 2.0;
    }

    // Current Option 2: (the manual road)
    recycle_std<kokkos_array> allocator;
    kokkos_array *input_array2 = allocator.allocate(1); // allocate memory
    allocator.construct(input_array2, std::string("my_manual_view")); // initialize kokkos view
    for (size_t i = 0; i < 100; i++) {
        input_array2[0](i) = i * 2.0;
    }
    allocator.destroy(input_array2);
    allocator.deallocate(input_array2, 1);

    // Current Option 3: (the vector-save-me road)
    // Problematic since it wants to initalize with the default view constructor and that one does actually not initialize the kokkos view
    std::vector<kokkos_array, recycle_std<kokkos_array>> input_array3(1);
    input_array3[0] = kokkos_array("my_vector_view"); //hence this line...
    for (size_t i = 0; i < 100; i++) {
        input_array3[0](i) = i * 2.0;
    }


    // These should use some of the recycled memory:
    {
        auto ptr = make_or_recycle_unique<kokkos_array>();
        auto ptr1 = make_or_recycle_unique<kokkos_array>();
        auto ptr2 = make_or_recycle_unique<kokkos_array>();
    }

    // These don't work properly because the default constructor of Kokkos::View is useless and does not initialize anything:
    {
        std::vector<kokkos_array, recycle_std<kokkos_array>> test1(1);
        std::vector<kokkos_array, recycle_std<kokkos_array>> test2(1);
    }
    {
        std::vector<kokkos_array, recycle_std<kokkos_array>> test1(1);
        std::vector<kokkos_array, recycle_std<kokkos_array>> test2(1);
        std::vector<kokkos_array, recycle_std<kokkos_array>> test3(1);
    }

}