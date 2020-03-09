
#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <hpx/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

#include "../include/buffer_manager.hpp"
#include <hpx/timing/high_resolution_timer.hpp>
#include <memory>

// scoped_timer -- stolen from Mikael
class [[nodiscard]] scoped_timer
{
public:
  scoped_timer(std::string const &label)
      : label(label), timer() {}
  ~scoped_timer()
  {
    std::ostringstream s;
    s << label << ": " << timer.elapsed() << " seconds" << std::endl;
    std::cerr << s.str();
  }

private:
  std::string label;
  hpx::util::high_resolution_timer timer;
};


//using kokkos_array = Kokkos::View<float[1000], Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
constexpr size_t view_size_0 = 10;
constexpr size_t view_size_1 = 50;
using type_in_view = float[view_size_1][view_size_0]; // todo deduce from kokkos_array
constexpr size_t view_size = view_size_0*view_size_1; // todo deduce from kokkos_array
using kokkos_array = Kokkos::View<type_in_view, Kokkos::HostSpace>;
// using kokkos_pinned_array = Kokkos::View<type_in_view, Kokkos::CudaHostPinnedSpace>;
// using kokkos_cuda_array = Kokkos::View<type_in_view, Kokkos::CudaSpace>;

//TODO put into own header
template <class kokkos_type, class alloc_type, class element_type>
class recycled_view : public kokkos_type {
    private:
        static alloc_type allocator;
        size_t total_elements;
    public:
        template <class... Args>
        recycled_view(Args... args) :
          kokkos_type(allocator.allocate(kokkos_type::required_allocation_size(args...) / sizeof(element_type)),args...),
          total_elements(kokkos_type::required_allocation_size(args...) / sizeof(element_type)) {
            //std::cout << "Got buffer for " << total_elements << std::endl;
        }
        ~recycled_view(void) {
            allocator.deallocate(this->data(), total_elements);
        }

  //TODO implement as appropriate
  explicit recycled_view(const recycled_view &other) = delete;

  explicit recycled_view(recycled_view &&other) noexcept = delete;

  recycled_view &operator=(const recycled_view &other) = delete;

  recycled_view &operator=(recycled_view &&other) noexcept = delete;
};

template <class kokkos_type, class alloc_type, class element_type>
alloc_type recycled_view<kokkos_type, alloc_type, element_type>::allocator;

// Just some 2D views used for testing
template <class T>
using kokkos_um_array = Kokkos::View<T**, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view = recycled_view<kokkos_um_array<T>, recycle_std<T>, T>;

template <class T>
using kokkos_um_device_array = Kokkos::View<T**, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_device_view = recycled_view<kokkos_um_device_array<T>, recycle_allocator_cuda_device<T>, T>;

template <class T>
using kokkos_um_pinned_array = Kokkos::View<T**, Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view = recycled_view<kokkos_um_pinned_array<T>, recycle_allocator_cuda_host<T>, T>;


/**
 * get an MDRangePolicy suitable for iterating the views
 * 
 * @param executor          a kokkos ExecutionSpace, e.g. hpx::kokkos::make_execution_space<Kokkos::Cuda>()
 * @param view_to_iterate   the view that needs to be iterated
 */
template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor& executor, const ViewType& view_to_iterate)
{
    constexpr auto rank = Kokkos::ViewTraits<type_in_view>::rank;
    const Kokkos::Array<int64_t, rank> zeros{};
    Kokkos::Array<int64_t, rank> extents;
    for (int i = 0; i < rank; ++i)
    {
        extents[i] = view_to_iterate.extent(i);
    }

  // //TODO what exactly does HintLightWeight do? cf. https://github.com/kokkos/kokkos/issues/1723
    // return Kokkos::Experimental::require(Kokkos::MDRangePolicy<decltype(executor), Kokkos::Rank<rank>>(executor,
    //                                                                                                    zeros, extents),
    //                                      Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    return Kokkos::MDRangePolicy<Executor, Kokkos::Rank<rank>>(executor, zeros, extents);
}

template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor&& executor, const ViewType& view_to_iterate){ 
    return get_iteration_policy(executor, view_to_iterate);
}

template <typename Viewtype>
struct kernel {
  const Viewtype & first;
  const Viewtype & second;
  Viewtype & output;

  kernel(const Viewtype & a, const Viewtype & b,
         Viewtype & c)
      : first(a), second(b), output(c) {}

  KOKKOS_INLINE_FUNCTION void operator()(std::size_t const j, std::size_t const k) const {
        printf("before all ");
        for (volatile int i = 0; i < 1000000;)
        {
          ++i;
        }
        printf("before assignment ");
        output(j, k) = first(j, k) + second(j, k);
  }
};

template <typename Viewtype, typename Policytype>
KOKKOS_INLINE_FUNCTION void kernel_add(const Viewtype &first, const Viewtype &second, Viewtype &output, const Policytype &policy)
{

  printf("outside all ");
  hpx::kokkos::parallel_for_async(
      "kernel add",
      policy,
        KOKKOS_LAMBDA(int j, int k) {
        // useless loop to make the computation last longer in the profiler
        printf("before all ");
        for (volatile int i = 0; i < 1000000;)
        {
          ++i;
        }
        printf("before assignment ");
        output(j, k) = first(j, k) + second(j, k);
        printf("after assignment ");
      });
}
template <typename Policytype>
KOKKOS_INLINE_FUNCTION void kernel_dummy(const Policytype &policy)
{

  printf("outside all ");
  hpx::kokkos::parallel_for_async(
      "kernel dummy",
      policy,
        KOKKOS_LAMBDA(int j) {
        // [&]__host__ __device__(int j) { 
          // todo: error: An extended __host__ __device__ lambda cannot capture variables by reference
        printf("before all ");
        for (volatile int i = 0; i < 1000000;)
        {
          ++i;
        }
        printf("after all ");
      });
}

void stream_executor_test()
{
  auto totalTimer = scoped_timer("total stream executor");

  const int numIterations = 40;
  static double d = 0;
  ++d;
  double t = d;

  recycled_host_view<double> hostView(view_size_0,view_size_1);

  {
    recycled_pinned_view<double> pinnedView(view_size_0,view_size_1);
    recycled_device_view<double> deviceView(view_size_0,view_size_1);

    auto policy_host = get_iteration_policy(Kokkos::DefaultHostExecutionSpace(), pinnedView);
    auto policy_host_manually = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(Kokkos::DefaultHostExecutionSpace(), {0, 0}, {view_size_0,view_size_1});

    static_assert(std::is_same<decltype(policy_host_manually),
                               decltype(policy_host)>::value);

    auto copy_finished = hpx::kokkos::parallel_for_async(
        "pinned host init",
        policy_host,
        [&](int n, int o) {
          hostView(n, o) = t;
          pinnedView(n, o) = hostView(n, o);
        });

    copy_finished.wait();

    // auto stream_space = hpx::kokkos::make_execution_space();
    auto stream_space = hpx::kokkos::make_execution_space<Kokkos::Cuda>(); // why do we only get the default stream??
    auto policy_stream = get_iteration_policy(stream_space, pinnedView);
    auto policy_stream_manually = Kokkos::MDRangePolicy<decltype(stream_space), Kokkos::Rank<2>>(stream_space, {0, 0}, {view_size_0, view_size_1});

    static_assert(std::is_same<decltype(policy_stream_manually),
                               decltype(policy_stream)>::value);

    {
      auto totalTimer = scoped_timer("async device");
      hpx::future<void> f;
      for (int i = 0; i < numIterations; ++i)
      {
        hpx::kokkos::deep_copy_async(stream_space, deviceView, pinnedView);

        kernel_dummy(hpx::kokkos::RangePolicy<>(stream_space, 0, i));
        // kernel_add(deviceView, deviceView, deviceView, policy_stream_manually);
        // auto kernel_functor = kernel<decltype(deviceView)>(deviceView, deviceView, deviceView);
        // hpx::kokkos::parallel_for_async(
        //   "kernel add",
        //   policy_stream_manually, 
        //   kernel_functor);

        f = hpx::kokkos::deep_copy_async(stream_space, pinnedView, deviceView);
      }
      f.wait();
    }

    hpx::kokkos::parallel_for_async(
        "pinned host copy back",
        policy_host,
        [&](int n, int o) {
          hostView(n, o) = pinnedView(n, o);
        })
        .wait();

    // test values in hostView
    // assert(hostView.data()[0] == t * (static_cast<unsigned long>(1) << numIterations));
          printf("%f hd ", hostView.data()[0]);
    Kokkos::fence();
  }
}

// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{
    hpx::kokkos::ScopeGuard g(argc, argv);

    /** Stress test for safe concurrency and performance:
   *  stolen from allocator_test
   * */

    constexpr size_t number_futures = 64;
    constexpr size_t passes = 10;

    static_assert(passes >= 0);
    assert(number_futures >= hpx::get_num_worker_threads());

    auto begin = std::chrono::high_resolution_clock::now();
    std::array<hpx::future<void>, number_futures> futs;
    for (size_t i = 0; i < number_futures; i++)
    {
        futs[i] = hpx::make_ready_future<void>();
    }
    for (size_t pass = 0; pass < passes; pass++)
    {
        for (size_t i = 0; i < number_futures; i++)
        {
            futs[i] = futs[i].then([&](hpx::future<void> &&predecessor) {
                stream_executor_test();
            });
        }
    }
    auto when = hpx::when_all(futs);
    when.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\n==>Allocation test took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
}
