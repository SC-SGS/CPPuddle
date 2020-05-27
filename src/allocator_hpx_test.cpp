
#include <chrono>
#include <cstdio>
#include <typeinfo>

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/program_options.hpp>

#include "../include/buffer_manager.hpp"

constexpr size_t max_number_futures = 64;
size_t number_futures = 64;
size_t array_size = 500000;
size_t passes = 200;

// #pragma nv_exec_check_disable
int hpx_main(int argc, char *argv[]) {
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
        "Sets the number of repetitions");

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
      return EXIT_SUCCESS;
    }
  } catch (const boost::program_options::error &ex) {
    std::cerr << "CLI argument problem found: " << ex.what() << '\n';
  }

  assert(passes >= 1);
  assert(array_size >= 1);
  assert(number_futures >= 1);
  assert(number_futures <= max_number_futures);

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
    std::cout << "\n==> Aggressive recycle allocation test took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
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
    std::cout << "\n==> Recycle allocation test took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "ms" << std::endl;
  }

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
    std::cout << "\n==> Non-recycle allocation test took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "ms" << std::endl;
  }
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
