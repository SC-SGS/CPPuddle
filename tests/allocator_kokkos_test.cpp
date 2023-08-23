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
#ifdef CPPUDDLE_HAVE_HPX  
#include <hpx/hpx_init.hpp>
#endif
#include <boost/program_options.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <boost/program_options.hpp>
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

#ifdef CPPUDDLE_HAVE_HPX
int hpx_main(int argc, char *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif

  std::string filename{};
  try {
    boost::program_options::options_description desc{"Options"};
    desc.add_options()("help", "Help screen")(
        "outputfile",
        boost::program_options::value<std::string>(&filename)->default_value(
            ""),
        "Redirect stdout/stderr to this file");

    boost::program_options::variables_map vm;
    boost::program_options::parsed_options options =
        parse_command_line(argc, argv, desc);
    boost::program_options::store(options, vm);
    boost::program_options::notify(vm);
    if (vm.count("help") == 0u) {
      std::cout << "Running with parameters:" << std::endl
                << " --filename = " << filename << std::endl;
    } else {
      std::cout << desc << std::endl;
      return EXIT_SUCCESS;
    }
  } catch (const boost::program_options::error &ex) {
    std::cerr << "CLI argument problem found: " << ex.what() << '\n';
  }
  if (!filename.empty()) {
    freopen(filename.c_str(), "w", stdout); // NOLINT
    freopen(filename.c_str(), "w", stderr); // NOLINT
  }


  hpx::kokkos::ScopeGuard scopeGuard(argc, argv);
  Kokkos::print_configuration(std::cout);

  using test_view = recycled_host_view<float>;
  using test_double_view = recycled_host_view<double>;

  constexpr size_t passes = 100;
  for (size_t pass = 0; pass < passes; pass++) {
    test_view my_wrapper_test1(1000);
    test_view my_wrapper_test2(1000);
    test_view my_wrapper_test3(recycler::number_instances - 1, 1000); // test 1D with location id 
    double t = 2.6;
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, 1000),
                        KOKKOS_LAMBDA(const int n) {
                            my_wrapper_test1.access(n) = t;
                            my_wrapper_test2.access(n) =
                                my_wrapper_test1.access(n);
                        });
    Kokkos::fence();
  }
  recycler::print_performance_counters();
#ifdef CPPUDDLE_HAVE_HPX  
  return hpx::finalize();
#else
  return EXIT_SUCCESS;
#endif
}

#ifdef CPPUDDLE_HAVE_HPX
int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
#endif
