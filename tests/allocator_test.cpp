// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef CPPUDDLE_HAVE_HPX  
#include <hpx/hpx_init.hpp>
#endif
#include <boost/program_options.hpp>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <typeinfo>

#include "std_recycling_allocators.hpp"

#ifdef CPPUDDLE_HAVE_HPX
int hpx_main(int argc, char *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif

  size_t array_size = 500000;
  size_t passes = 10000;
  std::string filename{};

  try {
    boost::program_options::options_description desc{"Options"};
    desc.add_options()("help", "Help screen")(
        "arraysize",
        boost::program_options::value<size_t>(&array_size)
            ->default_value(5000000),
        "Size of the buffers")(
        "passes",
        boost::program_options::value<size_t>(&passes)->default_value(200),
        "Sets the number of repetitions")(
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
                << " --arraysize = " << array_size << std::endl
                << " --passes = " << passes << std::endl;
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

  assert(passes >= 1);     // NOLINT
  assert(array_size >= 1); // NOLINT

  size_t aggressive_duration = 0;
  size_t recycle_duration = 0;
  size_t default_duration = 0;

  // Aggressive recycle Test:
  {
    std::cout << "\nStarting run with aggressive recycle allocator: " << std::endl;
    for (size_t pass = 0; pass < passes; pass++) {
      auto begin = std::chrono::high_resolution_clock::now();
      std::vector<double,
                  cppuddle::memory_recycling::aggressive_recycle_std<double>>
      test1(array_size, double{});
      auto end = std::chrono::high_resolution_clock::now();
      aggressive_duration +=
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count();
      // Print last element - Causes the compiler to not optimize out the entire loop
      std::cout << test1[array_size - 1] << " "; 
    }
    std::cout << "\n\n==> Aggressive recycle allocation test took "
              << aggressive_duration << "ms" << std::endl;
  }
  cppuddle::memory_recycling::print_buffer_counters();
  cppuddle::memory_recycling::force_buffer_cleanup(); // Cleanup all buffers and the managers for
                                    // better comparison

  // Recycle Test:
  {
    std::cout << "\nStarting run with recycle allocator: " << std::endl;
    for (size_t pass = 0; pass < passes; pass++) {
      auto begin = std::chrono::high_resolution_clock::now();
      std::vector<double, cppuddle::memory_recycling::recycle_std<double>>
        test1(array_size, double{});
      auto end = std::chrono::high_resolution_clock::now();
      recycle_duration +=
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count();
      // Print last element - Causes the compiler to not optimize out the entire loop
      std::cout << test1[array_size - 1] << " "; 
    }
    std::cout << "\n\n==> Recycle allocation test took " << recycle_duration
              << "ms" << std::endl;
  }
  cppuddle::memory_recycling::print_buffer_counters();
  cppuddle::memory_recycling::force_buffer_cleanup(); // Cleanup all buffers and the managers for
                                    // better comparison

  // Same test using std::allocator:
  {
    std::cout << "\nStarting run with std::allocator: " << std::endl;
    for (size_t pass = 0; pass < passes; pass++) {
      auto begin = std::chrono::high_resolution_clock::now();
      std::vector<double> test2(array_size, double{});
      auto end = std::chrono::high_resolution_clock::now();
      default_duration +=
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count();
      // Print last element - Causes the compiler to not optimize out the entire loop
      std::cout << test2[array_size - 1] << " "; 
    }
    std::cout << "\n\n==> Non-recycle allocation test took " << default_duration
              << "ms" << std::endl;
  }

  if (aggressive_duration < recycle_duration) {
    std::cout << "Test information: Aggressive recycler was faster than normal "
                 "recycler!"
              << std::endl;
  }
  if (aggressive_duration < default_duration) {
    std::cout << "Test information: Aggressive recycler was faster than default allocator!"
              << std::endl;
  }
  cppuddle::memory_recycling::print_buffer_counters();
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
