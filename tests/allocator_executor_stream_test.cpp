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
#include <hpx/timing/high_resolution_timer.hpp>
#include <memory>

// scoped_timer -- stolen from Mikael
class [[nodiscard]] scoped_timer {
public:
  explicit scoped_timer(const std::string &label) : label(std::move(label)) {}
  scoped_timer(scoped_timer &&) = default;
  scoped_timer &operator=(scoped_timer &&) = default;
  scoped_timer(const scoped_timer &) = default;
  scoped_timer &operator=(const scoped_timer &) = default;
  ~scoped_timer() {
    std::ostringstream s;
    s << label << ": " << timer.elapsed() << " seconds" << std::endl;
    std::cerr << s.str();
  }

private:
  std::string label;
  hpx::chrono::high_resolution_timer timer;
};

constexpr size_t view_size_0 = 10;
constexpr size_t view_size_1 = 50;
constexpr size_t view_size = view_size_0 * view_size_1;

template <class T>
using recycled_host_vector = std::vector<T, recycler::recycle_std<T>>;
template <class T>
using recycled_device_vector =
    std::vector<T, recycler::recycle_allocator_cuda_device<T>>;
template <class T>
using recycled_pinned_vector =
    std::vector<T, recycler::recycle_allocator_cuda_host<T>>;

void stream_executor_test() {
  auto totalTimer = scoped_timer("total stream executor");
  static double d = 0;
  ++d;

  recycled_host_vector<double> hostVector(view_size);
  recycled_pinned_vector<double> pinnedVector(view_size);

  // TODO(pollinta) this here segfaults for me, why?
  recycled_device_vector<double> deviceVector(view_size);

  // TODO(pollinta) reference in __host__ __device__ lambda problem
  auto g1 = hpx::experimental::for_loop(
      hpx::execution::par(hpx::execution::task), 0,
      view_size,
      // [&hostVector] HPX_HOST_DEVICE (std::size_t i) { hostVector[i] =
      // std::sin(double(i)); });
      [&hostVector](std::size_t i) { hostVector[i] = std::sin(double(i)); });

  // auto g2 = hpx::parallel::for_loop(
  //     hpx::parallel::execution::par(hpx::parallel::execution::task).on(hpx::compute::cuda::default_executor(t)),
  //     0, n,
  //     [&deviceVector] HPX_HOST_DEVICE (std::size_t i) { deviceVector[i] =
  //     std::sin(double(i)); });

  // TODO(pollinta) is there a way to use stream synchronization with HPX-only
  // executors too?

  g1.wait();
}

// #pragma nv_exec_check_disable
int main(int argc, char *argv[]) {
  hpx::kokkos::ScopeGuard g(argc, argv);

  /** Stress test for safe concurrency and performance:
   *  stolen from allocator_test
   * */

  constexpr size_t number_futures = 64;
  constexpr size_t passes = 10;

  static_assert(passes >= 0);
  assert(number_futures >= hpx::get_num_worker_threads());

  auto begin = std::chrono::high_resolution_clock::now();
  std::array<hpx::shared_future<void>, number_futures> futs;
  for (size_t i = 0; i < number_futures; i++) {
    futs[i] = hpx::make_ready_future<void>();
  }
  for (size_t pass = 0; pass < passes; pass++) {
    for (size_t i = 0; i < number_futures; i++) {
      futs[i] = futs[i].then([&](hpx::shared_future<void> &&predecessor) {
        stream_executor_test();
      });
    }
  }
  auto when = hpx::when_all(futs);
  when.wait();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "\n==>Allocation test took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "ms" << std::endl;
  recycler::force_cleanup();
}
