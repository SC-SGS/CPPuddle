
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

// scoped_timer -- stolen from Mikael
class [[nodiscard]] scoped_timer {
public:
  explicit scoped_timer(const std::string &label) : label(label) {}
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
  hpx::util::high_resolution_timer timer;
};

// using kokkos_array = Kokkos::View<float[1000], Kokkos::HostSpace,
// Kokkos::MemoryUnmanaged>;
constexpr size_t view_size_0 = 10;
constexpr size_t view_size_1 = 50;
using type_in_view = float[view_size_1][view_size_0];
constexpr size_t view_size = view_size_0 * view_size_1;
using kokkos_array = Kokkos::View<type_in_view, Kokkos::HostSpace>;
// using kokkos_pinned_array = Kokkos::View<type_in_view,
// Kokkos::CudaHostPinnedSpace>; using kokkos_cuda_array =
// Kokkos::View<type_in_view, Kokkos::CudaSpace>;

// Just some 2D views used for testing
template <class T>
using kokkos_um_device_array =
    Kokkos::View<T **, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_device_view =
    recycler::recycled_view<kokkos_um_device_array<T>,
                            recycler::recycle_allocator_cuda_device<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
template <class T>
using kokkos_um_array =
    Kokkos::View<T **, typename kokkos_um_device_array<T>::array_layout,
                 Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view =
    recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
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

template <typename Viewtype, typename Policytype>
KOKKOS_INLINE_FUNCTION void
kernel_add_kokkos(const Viewtype &first, const Viewtype &second,
                  Viewtype &output, const Policytype &policy) {
  Kokkos::parallel_for(
      "kernel add", policy, KOKKOS_LAMBDA(int j, int k) {
        // useless loop to make the computation last longer in the profiler
        for (volatile double i = 0.; i < 100.;) {
          ++i;
        }
        output(j, k) = first(j, k) + second(j, k);
      });
}

void stream_executor_test() {
  // auto totalTimer = scoped_timer("total stream executor");

  const int numIterations = 40;
  static double d = 0;
  ++d;
  double t = d;

  recycled_host_view<double> hostView(view_size_0, view_size_1);
  recycled_pinned_view<double> pinnedView(view_size_0, view_size_1);
  recycled_device_view<double> deviceView(view_size_0, view_size_1);

  {
    auto host_space =
        hpx::kokkos::make_execution_space<Kokkos::DefaultHostExecutionSpace>();
    auto policy_host = get_iteration_policy(host_space, pinnedView);

    auto copy_finished = hpx::kokkos::parallel_for_async(
        "pinned host init", policy_host, KOKKOS_LAMBDA(int n, int o) {
          hostView(n, o) = t;
          pinnedView(n, o) = hostView(n, o);
        });

    // auto stream_space = hpx::kokkos::make_execution_space();
    auto stream_space = hpx::kokkos::make_execution_space<Kokkos::Cuda>();
    auto policy_stream = get_iteration_policy(stream_space, pinnedView);

    // TODO(pollinta): How to make a nice continuation from HPX future to CUDA
    // stream (i.e. without using wait)?
    copy_finished.wait();

    // All of the following deep copies and kernels are sequenced because they
    // use the same instance. It is enough to wait for the last future. // The
    // views must have compatible layouts to actually use cudaMemcpyAsync.
    hpx::kokkos::deep_copy_async(stream_space, deviceView, pinnedView);

    {
      // auto totalTimer = scoped_timer("async device");
      for (int i = 0; i < numIterations; ++i) {
        kernel_add_kokkos(deviceView, deviceView, deviceView, policy_stream);
      }
    }

    hpx::kokkos::deep_copy_async(stream_space, pinnedView, deviceView);
    hpx::kokkos::deep_copy_async(stream_space, hostView, pinnedView).wait();

    // test values in hostView
    // printf("%f %f hd ", hostView.data()[0], t);
    assert(std::abs(hostView.data()[0] - t * (static_cast<unsigned long>(1)
                                              << numIterations)) < 1e-6);
  }
}

// #pragma nv_exec_check_disable
int main(int argc, char *argv[]) {
  hpx::kokkos::ScopeGuard g(argc, argv);

  /** Stress test for safe concurrency and performance:
   *  stolen from allocator_test
   * */

  constexpr size_t number_futures = 64;
  constexpr size_t passes = 5;

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
  hpx::wait_all(futs);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "\n==>Allocation test took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "ms" << std::endl;
}
