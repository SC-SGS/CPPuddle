#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

// scoped_timer -- stolen from Mikael
#include <hpx/timing/high_resolution_timer.hpp>

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

struct hello_world
{
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const
  {
    printf("Hello from i = %i\n", i);
  }
};

template <typename Viewtype>
void printSomeContents(Viewtype &printView)
{
  Kokkos::parallel_for("print some contents",
                       Kokkos::MDRangePolicy<typename Viewtype::execution_space, Kokkos::Rank<2>>(
                           {0, 0}, {printView.extent(0), printView.extent(1)}),
                       KOKKOS_LAMBDA(int j, int k) {
                         if ((j * k) % 1000 == 1)
                         {
                           printf("%d,%d , %f; ", j, k, printView(j, k));
                         }
                       });
}

Kokkos::View<double **, Kokkos::Experimental::HPX::array_layout, Kokkos::Cuda::memory_space>
initializeAndMirrorToDevice(Kokkos::View<double **, Kokkos::HostSpace> &testView)
{
  Kokkos::parallel_for("init",
                       Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
                           {0, 0}, {testView.extent(0), testView.extent(1)}),
                       KOKKOS_LAMBDA(int j, int k) {
                         volatile double a = std::pow(j * k, 2);
                         testView(j, k) = a;
                       });
  Kokkos::fence();
  printSomeContents(testView);

  auto deviceMirror = Kokkos::create_mirror_view(typename Kokkos::DefaultExecutionSpace::memory_space(), testView);

  Kokkos::deep_copy(deviceMirror, testView);
  Kokkos::fence();
  return deviceMirror;
}

template <typename Viewtype>
double getSumOfViewEntries(Viewtype &reduceView)
{
  double reductionResult = 0;
  Kokkos::parallel_reduce("reduce all",
                          Kokkos::MDRangePolicy<typename Viewtype::execution_space, Kokkos::Rank<2>>(
                              // Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                              {0, 0}, {reduceView.extent(0), reduceView.extent(1)}),
                          KOKKOS_LAMBDA(int j, int k, double &sum) {
                            sum += reduceView(j, k);
                          },
                          reductionResult);
  // Kokkos::fence();
  return reductionResult;
}

Kokkos::View<double **, Kokkos::Cuda::array_layout, Kokkos::HostSpace> initializeAndMirrorToHost(Kokkos::View<double **, Kokkos::DefaultExecutionSpace> &testView)
{
  Kokkos::parallel_for("init",
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                           {0, 0}, {testView.extent(0), testView.extent(1)}),
                       KOKKOS_LAMBDA(int j, int k) {
                         testView(j, k) = j * k;
                       });
  Kokkos::fence();
  printSomeContents(testView);

  auto hostMirror = Kokkos::create_mirror_view(testView);

  Kokkos::deep_copy(hostMirror, testView);
  Kokkos::fence();
  return hostMirror;
}

template <typename Viewtype, typename Policytype>
void kernel_add(Viewtype &first, Viewtype &second, Viewtype &output, Policytype &policy)
{
  Kokkos::parallel_for(
      "kernel add",
      policy,
      KOKKOS_LAMBDA(int j, int k) {
        output(j, k) = first(j, k) + second(j, k);
      });
}

struct CudaStreamView
{
  cudaStream_t stream_;
  Kokkos::Cuda cuda_;
  float *p_, *q_, *r_;
  Kokkos::View<float **, Kokkos::CudaSpace> view_p, view_q, view_r;
  decltype(Kokkos::create_mirror_view(view_p)) host_p, host_q, host_r;
  // //TODO deduce this type; problem: no default constructor
  // Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Schedule<Kokkos::Static>, void,
  //                       Kokkos::Cuda::size_type, Kokkos::Rank<2U, Kokkos::Iterate::Default,
  //                                                             Kokkos::Iterate::Default>,
  //                       Kokkos::LaunchBounds<0U, 0U>,
  //                       Kokkos::Experimental::WorkItemProperty::ImplWorkItemProperty<1UL>>
  //                       iterate_p_policy;

  CudaStreamView(const std::size_t j, const std::size_t k)
  {
    // set up cuda stream
    cudaStreamCreate(&stream_);
    cuda_ = Kokkos::Cuda(stream_);

    // create and initialize device views and their HostMirrors
    // cf https://github.com/kokkos/kokkos/blob/master/core/unit_test/cuda/TestCuda_InterOp_Streams.cpp#L137

    const std::size_t N = j * k;

    cudaMalloc(&p_, static_cast<std::size_t>(sizeof(float) * N));
    cudaMalloc(&q_, static_cast<std::size_t>(sizeof(float) * N));
    cudaMalloc(&r_, static_cast<std::size_t>(sizeof(float) * N));

    view_p = Kokkos::View<float **, Kokkos::CudaSpace>(p_, j, k);
    view_q = Kokkos::View<float **, Kokkos::CudaSpace>(q_, j, k);
    view_r = Kokkos::View<float **, Kokkos::CudaSpace>(r_, j, k);
    Kokkos::deep_copy(cuda_, view_p, 5.);
    Kokkos::deep_copy(cuda_, view_q, 6.);
    Kokkos::deep_copy(cuda_, view_r, 7.);

    cuda_.fence();

    host_p = Kokkos::create_mirror_view(view_p);
    host_q = Kokkos::create_mirror_view(view_q);
    host_r = Kokkos::create_mirror_view(view_r);
    Kokkos::deep_copy(host_p, view_p);
    Kokkos::deep_copy(host_q, view_q);
    Kokkos::deep_copy(host_r, view_r);

    // // create policy for iterating Views
    // // cf https://github.com/kokkos/kokkos/issues/1723#issuecomment-464281156
    // iterate_p_policy = Kokkos::Experimental::require( Kokkos::MDRangePolicy<Kokkos::Rank<2>>(cuda_,
    //       {0, 0}, {view_p.extent(0), view_p.extent(1)}), Kokkos::Experimental::WorkItemProperty::HintLightWeight );

    cuda_.fence();
  }

  ~CudaStreamView()
  {
    cudaFree(p_);
    cudaFree(q_);
    cudaFree(r_);
  }
};

void cuda_small_kernel_test()
{
  {
    auto totalTimer = scoped_timer("total small kernels");

    // set up cuda streams
    const int numStreams = 5;

    const std::size_t j = 100;
    const std::size_t k = 100;

    CudaStreamView cuStreamView(j, k);
    CudaStreamView cuStreamView2(j, k);
    CudaStreamView cuStreamView3(j, k);
    CudaStreamView cuStreamView4(j, k);
    CudaStreamView cuStreamView5(j, k);

    const std::array<CudaStreamView, numStreams> CuViews{cuStreamView, cuStreamView2, cuStreamView3, cuStreamView4, cuStreamView5};

    Kokkos::fence();

    // // launch data-independent kernel
    // {
    //   auto sTimer = scoped_timer("kernel nodata");
    //   for (int i = 0; i < numStreams; ++i)
    //   {
    //     auto iterate_p_policy = Kokkos::Experimental::require(Kokkos::MDRangePolicy<Kokkos::Rank<cuStreamView.view_p.rank>>(cuStreamView.cuda_,
    //                                                                                                                         {0, 0}, {cuStreamView.view_p.extent(0), cuStreamView.view_p.extent(1)}),
    //                                                           Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    //     for (int l = 0; l < 10000; ++l)
    //     {
    //       Kokkos::parallel_for("print some numbers",
    //                            iterate_p_policy,
    //                            KOKKOS_LAMBDA(int j, int k) {
    //                              if ((j * k) % 10000 == 1)
    //                              {
    //                                printf("%d,%d; ", j, k);
    //                              }
    //                            });
    //     }
    //   }
    //   Kokkos::fence();
    // }

    // launch data-dependent kernels
    {
      auto sTimer = scoped_timer("kernel add one stream");
      auto iterate_p_policy = Kokkos::Experimental::require(Kokkos::MDRangePolicy<Kokkos::Rank<cuStreamView.view_p.rank>>(cuStreamView.cuda_,
                                                                                                                          {0, 0}, {cuStreamView.view_p.extent(0), cuStreamView.view_p.extent(1)}),
                                                            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
      for (int i = 0; i < 10000; ++i)
      {
        kernel_add(cuStreamView.view_p, cuStreamView.view_q, cuStreamView.view_r, iterate_p_policy);
        cuStreamView.cuda_.fence();
        kernel_add(cuStreamView.view_p, cuStreamView.view_r, cuStreamView.view_q, iterate_p_policy);
        cuStreamView.cuda_.fence();
      }
      Kokkos::fence();
    }
    {
      auto sTimer = scoped_timer("kernel add Kokkos fence");
      for (const auto &csv : CuViews)
      {
        auto iterate_p_policy = Kokkos::Experimental::require(Kokkos::MDRangePolicy<Kokkos::Rank<cuStreamView.view_p.rank>>(csv.cuda_,
                                                                                                                            {0, 0}, {csv.view_p.extent(0), csv.view_p.extent(1)}),
                                                              Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        for (int i = 0; i < 10000; ++i)
        {
          kernel_add(csv.view_p, csv.view_q, csv.view_r, iterate_p_policy);
          Kokkos::fence();
          kernel_add(csv.view_p, csv.view_r, csv.view_q, iterate_p_policy);
          Kokkos::fence();
        }
      }
      Kokkos::fence();
    }
    {
      auto sTimer = scoped_timer("kernel add no fence");
      for (const auto &csv : CuViews)
      {
        auto iterate_p_policy = Kokkos::Experimental::require(Kokkos::MDRangePolicy<Kokkos::Rank<cuStreamView.view_p.rank>>(csv.cuda_,
                                                                                                                            {0, 0}, {csv.view_p.extent(0), csv.view_p.extent(1)}),
                                                              Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        for (int i = 0; i < 10000; ++i)
        {
          kernel_add(csv.view_p, csv.view_q, csv.view_r, iterate_p_policy);
          kernel_add(csv.view_p, csv.view_r, csv.view_q, iterate_p_policy);
        }
      }
      Kokkos::fence();
    }
    {
      auto sTimer = scoped_timer("kernel add deep_copy");
      for (const auto &csv : CuViews)
      {
        auto iterate_p_policy = Kokkos::Experimental::require(Kokkos::MDRangePolicy<Kokkos::Rank<cuStreamView.view_p.rank>>(csv.cuda_,
                                                                                                                            {0, 0}, {csv.view_p.extent(0), csv.view_p.extent(1)}),
                                                              Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        for (int i = 0; i < 10000; ++i)
        {
          kernel_add(csv.view_p, csv.view_q, csv.view_r, iterate_p_policy);
          Kokkos::deep_copy(csv.host_r, csv.view_r);
          kernel_add(csv.view_p, csv.view_r, csv.view_q, iterate_p_policy);
          Kokkos::deep_copy(csv.host_q, csv.view_q);
        }
      }
      Kokkos::fence();
    }

    printSomeContents(cuStreamView.host_r);

    Kokkos::fence();
  }
}

// #pragma nv_exec_check_disable
int main (int argc, char* argv[]) {
  Kokkos::initialize (argc, argv);

  Kokkos::print_configuration(std::cout);

  // cuda small kernel test, to emulate octotiger requirements
  // TODO HPX-ify
  {
    cuda_small_kernel_test();
    Kokkos::fence();
  }
  // Kokkos hello world example
  {
    std::size_t worker_id = hpx::get_worker_thread_num();
    std::size_t locality_id = hpx::get_locality_id();

    printf ("HPX Thread %i on Locality %i\n\n", worker_id, locality_id);

    Kokkos::parallel_for ("HelloWorld",Kokkos::RangePolicy<Kokkos::Cuda>(0, 14), hello_world ());
    Kokkos::fence();

    printf ("Hello World on Kokkos execution space %s\n",
            typeid (Kokkos::Experimental::HPX).name ());
    Kokkos::parallel_for ("HelloWorld",Kokkos::RangePolicy<Kokkos::Experimental::HPX>(0,14), hello_world ());
    Kokkos::fence();

    printf ("Hello World on Kokkos execution space %s\n",
            typeid (Kokkos::Serial).name ());
    Kokkos::parallel_for ("HelloWorld",Kokkos::RangePolicy<Kokkos::Serial>(0,14), hello_world ());
    Kokkos::fence();
  }
  // mirroring views and wrapping calls in HPX futures
  {
    // Mirror a Host view to device
    Kokkos::View<double**, Kokkos::HostSpace> testViewHost(
      Kokkos::ViewAllocateWithoutInitializing("test host"), 150, 560);
    auto deviceMirror = initializeAndMirrorToDevice(testViewHost);

    auto f = hpx::async([deviceMirror]() -> double {
                    printf ("\n reduce on device\n"); 
                    return getSumOfViewEntries(deviceMirror); 
                  });
    auto g = hpx::async([testViewHost]() -> double {
                    printf ("\n reduce on host\n"); 
                    return getSumOfViewEntries(testViewHost); 
                  });

    Kokkos::fence();

    // Mirror a device view to host
    printf("create default execution space view \n ");
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> testViewDevice(
		  Kokkos::ViewAllocateWithoutInitializing("state"), 150, 560);
    auto hostMirror = initializeAndMirrorToHost(testViewDevice);
    Kokkos::fence();
                  
    auto when = hpx::when_all(f, g);
    when.wait();
    // terminate called after throwing an instance of 'hpx::detail::exception_with_info<hpx::exception>'
    //  what():  this future has no valid shared state: HPX(no_state) =>why? these here seem to work:
    // f.wait();
    // g.wait();

    printf("Reduction results: %lf %lf \n", g.get(), f.get());

    Kokkos::fence();
  }
  Kokkos::finalize(); 
}