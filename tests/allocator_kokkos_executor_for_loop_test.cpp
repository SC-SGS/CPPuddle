// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define USE_HPX_MAIN
#ifdef USE_HPX_MAIN
#include <hpx/hpx_init.hpp>
#else
#include <hpx/hpx_main.hpp>
#endif
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <hpx/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"
#include "../include/kokkos_buffer_util.hpp"
#include <hpx/timing/high_resolution_timer.hpp>
#include <memory>

// Assert during Release builds as well for this file:
#undef NDEBUG
#include <cassert> // reinclude the header to update the definition of assert()

constexpr size_t view_size_0 = 10;
constexpr size_t view_size_1 = 50;
template <class T>

// Host views using recycle allocators
using kokkos_um_array =
    Kokkos::View<T **, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view =
    recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;


// Device views using recycle allocators
template <class T>
using kokkos_um_device_array =
    Kokkos::View<T **, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_device_view =
    recycler::recycled_view<kokkos_um_device_array<T>,
                            recycler::recycle_allocator_cuda_device<T>, T>;

// Host views using pinned memory recycle allocators
template <class T>
using kokkos_um_pinned_array =
    Kokkos::View<T **, typename kokkos_um_device_array<T>::array_layout,
                 Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>,
                            recycler::recycle_allocator_cuda_host<T>, T>;

template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor &&executor,
                          const ViewType &view_to_iterate) {
  return get_iteration_policy(executor, view_to_iterate);
}

#ifdef USE_HPX_MAIN
int hpx_main(int argc, char *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
  hpx::kokkos::ScopeGuard g(argc, argv);

  // otherwise the HPX cuda polling futures won't work
  hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));


  constexpr size_t passes = 100;

  // Host run
  for (size_t pass = 0; pass < passes; pass++) {
    // Create view
    recycled_host_view<double> hostView(view_size_0, view_size_1);

    // Create executor
    hpx::kokkos::serial_executor executor;

    // Obtain execution policy from executor
    auto policy_1 = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<2>>(
        executor.instance(), {0, 0}, {view_size_0, view_size_1}),
         Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    
    // Run with the execution policy
    Kokkos::parallel_for(
        "host init", policy_1,
        KOKKOS_LAMBDA(int n, int o) { hostView(n, o) = 1.0; });
    Kokkos::fence();
    
    // Verify
    for(size_t i = 0; i < view_size_0; i++) {
      for(size_t j = 0; j < view_size_1; j++) {
        assert(hostView(i, j) == 1.0);
      }
    }
  }

  // Device run
  for (size_t pass = 0; pass < passes; pass++) {
    // Create and init host view
    recycled_pinned_view<double> hostView(view_size_0, view_size_1);
    for(size_t i = 0; i < view_size_0; i++) {
      for(size_t j = 0; j < view_size_1; j++) {
        hostView(i, j) = 1.0;
      }
    }

    // Create executor
    hpx::kokkos::cuda_executor executor(hpx::kokkos::execution_space_mode::independent);

    // Use executor to move the host data to the device
   recycled_device_view<double> deviceView(view_size_0, view_size_1);
   Kokkos::deep_copy(executor.instance(), deviceView, hostView); 

    auto policy_1 = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<2>>(
        executor.instance(), {0, 0}, {view_size_0, view_size_1}),
         Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for(
        "device run", policy_1,
        KOKKOS_LAMBDA(int n, int o) { deviceView(n, o) = 2,0; });
        
   // No Kokkos::fence() required
   auto fut = hpx::kokkos::deep_copy_async(executor.instance(), hostView, deviceView); 
   fut.get(); 
    for(size_t i = 0; i < view_size_0; i++) {
      for(size_t j = 0; j < view_size_1; j++) {
        assert(hostView(i, j) == 2.0);
      }
    }
  }

  // otherwise the HPX cuda polling futures won't work
  hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
  // Cleanup all cuda views 
  // (otherwise the cuda driver might shut down before this gets done automatically at
  // the end of the programm)
  recycler::force_cleanup();
  return hpx::finalize();
}

#ifdef USE_HPX_MAIN
int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
#endif
