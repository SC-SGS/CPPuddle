#include "../include/buffer_manager.hpp"
#include <boost/program_options.hpp>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <typeinfo>

int main(int argc, char *argv[]) {

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

  // Aggressive recycle Test:
  {
    auto begin = std::chrono::high_resolution_clock::now();
    for (size_t pass = 0; pass < passes; pass++) {
      std::vector<double, recycler::aggressive_recycle_std<double>> test1(
          array_size, double{});
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\n==> Aggressive recycle allocation test took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "ms" << std::endl;
  }
  recycler::force_cleanup(); // Cleanup all buffers and the managers for better
                             // comparison

  // Recycle Test:
  {
    auto begin = std::chrono::high_resolution_clock::now();
    for (size_t pass = 0; pass < passes; pass++) {
      std::vector<double, recycler::recycle_std<double>> test1(array_size,
                                                               double{});
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\n==> Recycle allocation test took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "ms" << std::endl;
  }
  recycler::force_cleanup(); // Cleanup all buffers and the managers for better
                             // comparison

  // Same test using std::allocator:
  {
    auto begin = std::chrono::high_resolution_clock::now();
    for (size_t pass = 0; pass < passes; pass++) {
      std::vector<double> test2(array_size, double{});
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\n==> Non-recycle allocation test took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "ms" << std::endl;
  }
  return EXIT_SUCCESS;
}
