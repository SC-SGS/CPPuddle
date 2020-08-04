
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

constexpr size_t view_size_0 = 10;
constexpr size_t view_size_1 = 50;
template <class T>
using kokkos_um_array =
    Kokkos::View<T **, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view =
    recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;

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
  constexpr size_t passes = 1;
  for (size_t pass = 0; pass < passes; pass++) {
    recycled_host_view<double> hostView(view_size_0, view_size_1);
    auto host_space =
        hpx::kokkos::make_execution_space<Kokkos::DefaultHostExecutionSpace>();
    auto policy_host = get_iteration_policy(host_space, hostView);
    Kokkos::parallel_for(
        "host init", policy_host,
        KOKKOS_LAMBDA(int n, int o) { hostView(n, o) = 1.0; });
  }
  return hpx::finalize();
}

#ifdef USE_HPX_MAIN
int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
#endif