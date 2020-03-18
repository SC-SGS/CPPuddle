
#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <hpx/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

// scoped_timer -- stolen from Mikael
#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"
#include "../include/kokkos_buffer_util.hpp"
#include <hpx/timing/high_resolution_timer.hpp>
#include <memory>

using kokkos_array = Kokkos::View<float[1000], Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
// using kokkos_pinned_array = Kokkos::View<type_in_view, Kokkos::CudaHostPinnedSpace>;
// using kokkos_cuda_array = Kokkos::View<type_in_view, Kokkos::CudaSpace>;

// Just some 2D views used for testing
template <class T>
using kokkos_um_array = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view = recycled_view<kokkos_um_array<T>, recycle_std<T>, T>;

template <class T>
using kokkos_um_device_array = Kokkos::View<T*, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_device_view = recycled_view<kokkos_um_device_array<T>, recycle_allocator_cuda_device<T>, T>;


// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{
    hpx::kokkos::ScopeGuard scopeGuard(argc, argv);
    Kokkos::print_configuration(std::cout);

    // // Way 1 to recycle heap buffer as well (manually)
    // recycle_std<float> alli;
    // float *my_recycled_data_buffer = alli.allocate(1000); // allocate memory
    // {
    //     kokkos_um_array<float> test_buffered(my_recycled_data_buffer, 1000);
    //     for (size_t i = 0; i < 1000; i++) {
    //         test_buffered.data()[i] = i * 2.0;
    //     }
    // }
    // alli.deallocate(my_recycled_data_buffer, 1000); 
    // size_t to_alloc = kokkos_um_array<float>::required_allocation_size(1000);
    // std::cout << "Actual required size: "  << to_alloc << std::endl; // Still a heap allocation!

    // // Way 2 for recycling 
    // test_view my_wrapper_test0(1000);
    // for (size_t i = 0; i < 1000; i++) {
    //     my_wrapper_test0.data()[i] = i * 2.0;
    // }

    using test_view = recycled_host_view<float>;
    using test_double_view = recycled_host_view<double>;
    test_view my_wrapper_test1(1000);
    test_view my_wrapper_test2(1000);
    double t = 2.6;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::Experimental::HPX>(0, 1000), KOKKOS_LAMBDA(const int n) {
          my_wrapper_test1.access(n) = t;
          my_wrapper_test2.access(n) = my_wrapper_test1.access(n);
        });

    // for some views on cuda data
    using test_device_view = recycled_device_view<float>;
    using test_device_double_view = recycled_device_view<double>;



}
