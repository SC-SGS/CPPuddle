// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <typeinfo>

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/program_options.hpp>

#include "../include/buffer_manager.hpp"

int hpx_main(int argc, char *argv[]) {

  constexpr size_t max_number_futures = 1024;
  size_t number_futures = 64;
  size_t array_size = 500000;
  size_t passes = 200;
  std::string filename{};

  try {
    boost::program_options::options_description desc{"Options"};
    desc.add_options()("help", "Help screen")(
        "arraysize",
        boost::program_options::value<size_t>(&array_size)
            ->default_value(5000000),
        "Size of the buffers")(
        "futures",
        boost::program_options::value<size_t>(&number_futures)
            ->default_value(64),
        "Sets the number of futures to be (potentially) executed in parallel")(
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
                << " --futures =  " << number_futures << std::endl
                << " --passes = " << passes << std::endl
                << " --hpx:threads = " << hpx::get_os_thread_count()
                << std::endl;
    } else {
      std::cout << desc << std::endl;
      return hpx::finalize();
    }
  } catch (const boost::program_options::error &ex) {
    std::cerr << "CLI argument problem found: " << ex.what() << '\n';
  }
  if (!filename.empty()) {
    freopen(filename.c_str(), "w", stdout); // NOLINT
    freopen(filename.c_str(), "w", stderr); // NOLINT
  }

  assert(passes >= 1);                          // NOLINT
  assert(array_size >= 1);                      // NOLINT
  assert(number_futures >= 1);                  // NOLINT
  assert(number_futures <= max_number_futures); // NOLINT

  {
    size_t aggressive_duration = 0;
    size_t recycle_duration = 0;
    size_t default_duration = 0;

    // Same test using std::allocator:
    {
      auto begin = std::chrono::high_resolution_clock::now();
      std::vector<hpx::shared_future<void>> futs(max_number_futures);
      for (size_t i = 0; i < max_number_futures; i++) {
        futs[i] = hpx::make_ready_future<void>();
      }
      for (size_t pass = 0; pass < passes; pass++) {
        for (size_t i = 0; i < number_futures; i++) {
          futs[i] = futs[i].then([&](hpx::shared_future<void> &&predecessor) {
            std::vector<double> test6(array_size, double{});
          });
        }
      }
      auto when = hpx::when_all(futs);
      when.wait();
      auto end = std::chrono::high_resolution_clock::now();
      default_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count();
      std::cout << "\n==> Non-recycle allocation test took " << default_duration
                << "ms" << std::endl;
    }

    {
      auto begin = std::chrono::high_resolution_clock::now();
      std::vector<hpx::shared_future<void>> futs(max_number_futures);
      for (size_t i = 0; i < max_number_futures; i++) {
        futs[i] = hpx::make_ready_future<void>();
      }
      for (size_t pass = 0; pass < passes; pass++) {
        for (size_t i = 0; i < number_futures; i++) {
          futs[i] = futs[i].then([&](hpx::shared_future<void> &&predecessor) {
            std::vector<double, recycler::recycle_std<double>> test6(array_size,
                                                                     double{});
          });
        }
      }
      auto when = hpx::when_all(futs);
      when.wait();
      auto end = std::chrono::high_resolution_clock::now();
      recycle_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count();
      std::cout << "\n==> Recycle allocation test took " << recycle_duration
                << "ms" << std::endl;
    }
    recycler::print_performance_counters();
    recycler::force_cleanup(); // Cleanup all buffers and the managers for better
                               // comparison


    // ensure that at least 4 buffers have to created for unit testing
    {
      std::vector<double, recycler::aggressive_recycle_std<double>> buffer1(
          array_size, double{});
      std::vector<double, recycler::aggressive_recycle_std<double>> buffer2(
          array_size, double{});
      std::vector<double, recycler::aggressive_recycle_std<double>> buffer3(
          array_size, double{});
      std::vector<double, recycler::aggressive_recycle_std<double>> buffer4(
          array_size, double{});
    }

    // Aggressive recycle Test:
    {
      auto begin = std::chrono::high_resolution_clock::now();
      std::vector<hpx::shared_future<void>> futs(max_number_futures);
      for (size_t i = 0; i < max_number_futures; i++) {
        futs[i] = hpx::make_ready_future<void>();
      }
      for (size_t pass = 0; pass < passes; pass++) {
        for (size_t i = 0; i < number_futures; i++) {
          futs[i] = futs[i].then([&](hpx::shared_future<void> &&predecessor) {
            std::vector<double, recycler::aggressive_recycle_std<double>> test6(
                array_size, double{});
          });
        }
      }
      auto when = hpx::when_all(futs);
      when.wait();
      auto end = std::chrono::high_resolution_clock::now();
      aggressive_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count();
      std::cout << "\n==> Aggressive recycle allocation test took "
                << aggressive_duration << "ms" << std::endl;
    }
    recycler::print_performance_counters();
    recycler::force_cleanup(); // Cleanup all buffers and the managers for better
                               // comparison


    if (aggressive_duration < recycle_duration) {
      std::cout << "Test information: Aggressive recycler was faster than normal "
                   "recycler!"
                << std::endl;
    }
    if (recycle_duration < default_duration) {
      std::cout << "Test information: Recycler was faster than default allocator!"
                << std::endl;
    }
  }
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
