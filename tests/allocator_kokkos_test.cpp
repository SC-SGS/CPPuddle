// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
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

using kokkos_array =
    Kokkos::View<float[1000], Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

// Just some 2D views used for testing
template <class T>
using kokkos_um_array =
    Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view =
    recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;

// #pragma nv_exec_check_disable
int main(int argc, char *argv[]) {
  hpx::kokkos::ScopeGuard scopeGuard(argc, argv);
  Kokkos::print_configuration(std::cout);

  using test_view = recycled_host_view<float>;
  using test_double_view = recycled_host_view<double>;

  constexpr size_t passes = 100;
  for (size_t pass = 0; pass < passes; pass++) {
    test_view my_wrapper_test1(1000);
    test_view my_wrapper_test2(1000);
    double t = 2.6;
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, 1000),
                        KOKKOS_LAMBDA(const int n) {
                            my_wrapper_test1.access(n) = t;
                            my_wrapper_test2.access(n) =
                                my_wrapper_test1.access(n);
                        });
    Kokkos::fence();
  }
}
