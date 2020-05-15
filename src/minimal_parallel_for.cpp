
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

constexpr size_t view_size_0 = 10;
constexpr size_t view_size_1 = 50;
template <class T>
using kokkos_um_array = Kokkos::View<T**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view = recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;


template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor&& executor, const ViewType& view_to_iterate){ 
    return get_iteration_policy(executor, view_to_iterate);
}

int main(int argc, char *argv[])
{
    constexpr size_t passes = 1;
    for (size_t pass = 0; pass < passes; pass++)
    {
      recycled_host_view<double> hostView(view_size_0,view_size_1);

      // works - usage counter goes up to 9 for the hostView
      // auto host_space = hpx::kokkos::make_execution_space<Kokkos::Serial>(); 

      // broken - usage counter goes up to 13 for the hostView - goes only down to 2
      auto host_space = hpx::kokkos::make_execution_space<Kokkos::DefaultHostExecutionSpace>();
      auto policy_host = get_iteration_policy(host_space, hostView);
      Kokkos::parallel_for(
          "host init",
          policy_host,
          KOKKOS_LAMBDA(int n, int o) {
            hostView(n, o) = 1.0;
          });
    }
}
