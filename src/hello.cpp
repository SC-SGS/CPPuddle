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
    scoped_timer(std::string const& label)
      : label(label)
      , timer() {}
    ~scoped_timer() {
        std::ostringstream s;
        s << label << ": " << timer.elapsed() << " seconds" << std::endl;
        std::cerr << s.str();
    }

private:
    std::string label;
    hpx::util::high_resolution_timer timer;
};


struct hello_world {
  KOKKOS_INLINE_FUNCTION
  void operator() (const int i) const {
    printf ("Hello from i = %i\n", i);
  }
};


template <typename Viewtype>
void printSomeContents(Viewtype& printView){
  Kokkos::parallel_for("print some contents",
      Kokkos::MDRangePolicy<typename Viewtype::execution_space, Kokkos::Rank<2>>(
          {0, 0}, {printView.extent(0), printView.extent(1)}),
      KOKKOS_LAMBDA(int j, int k) {
        if((j*k)%1000 == 1){
          printf("%d,%d , %f; ", j, k, printView(j, k));
        }
      });
}

Kokkos::View<double**, Kokkos::Experimental::HPX::array_layout, Kokkos::Cuda::memory_space> 
  initializeAndMirrorToDevice(Kokkos::View<double**, Kokkos::HostSpace>& testView){
    Kokkos::parallel_for("init",
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {testView.extent(0), testView.extent(1)}),
        KOKKOS_LAMBDA(int j, int k) {
			    volatile double a = std::pow(j*k, 2);
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
double getSumOfViewEntries(Viewtype& reduceView){
  double reductionResult = 0;
  Kokkos::parallel_reduce("reduce all",
    Kokkos::MDRangePolicy<typename Viewtype::execution_space, Kokkos::Rank<2>>(
    // Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {0, 0}, {reduceView.extent(0), reduceView.extent(1)}),
    KOKKOS_LAMBDA(int j, int k,  double& sum) {
      sum += reduceView(j, k);
    }, reductionResult);
// Kokkos::fence();
  return reductionResult;
}

Kokkos::View<double**, Kokkos::Cuda::array_layout, Kokkos::HostSpace> initializeAndMirrorToHost
      (Kokkos::View<double**, Kokkos::DefaultExecutionSpace>& testView){
  Kokkos::parallel_for("init",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          {0, 0}, {testView.extent(0), testView.extent(1)}),
      KOKKOS_LAMBDA(int j, int k) {
          testView(j, k) = j*k;
      });
  Kokkos::fence();
  printSomeContents(testView);

  auto hostMirror = Kokkos::create_mirror_view(testView);

  Kokkos::deep_copy(hostMirror, testView);
	Kokkos::fence();
  return hostMirror;
}

template <typename Viewtype, typename Policytype>
void kernel_add(Viewtype& first, Viewtype& second, Viewtype& output, Policytype& policy){
  Kokkos::parallel_for(
    "kernel add",
    policy,
    KOKKOS_LAMBDA(int j, int k) {
        output(j, k) = first(j,k) + second(j,k);
    });
}

void cuda_small_kernel_test(){
  {  
    auto totalTimer = scoped_timer("total small kernels");

    // set up cuda streams
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    Kokkos::Cuda cuda1(stream1);
    Kokkos::Cuda cuda2(stream2);

    // create device views and their HostMirrors
    // cf https://github.com/kokkos/kokkos/blob/master/core/unit_test/cuda/TestCuda_InterOp_Streams.cpp#L137

    const std::size_t j = 100;
    const std::size_t k = 100;
    const std::size_t N = j*k;

    float* p, * q, * r;
    cudaMalloc(&p, static_cast<std::size_t>(sizeof(float)*N));
    cudaMalloc(&q, static_cast<std::size_t>(sizeof(float)*N));
    cudaMalloc(&r, static_cast<std::size_t>(sizeof(float)*N));

    Kokkos::View<float**,Kokkos::CudaSpace> view_p(p,j,k);
    Kokkos::View<float**,Kokkos::CudaSpace> view_q(q,j,k);
    Kokkos::View<float**,Kokkos::CudaSpace> view_r(r,j,k);
    Kokkos::deep_copy(cuda1,view_p,5.);
    Kokkos::deep_copy(cuda1,view_q,6.);
    Kokkos::deep_copy(cuda1,view_r,7.);

    auto host_p = Kokkos::create_mirror_view(view_p);
    auto host_q = Kokkos::create_mirror_view(view_q);
    auto host_r = Kokkos::create_mirror_view(view_r);

    // create policies for iterating Views
    // cf https://github.com/kokkos/kokkos/issues/1723#issuecomment-464281156

    auto iterate_p_policy_1 = Kokkos::Experimental::require( Kokkos::MDRangePolicy<Kokkos::Rank<view_p.rank>>(cuda1, 
            {0, 0}, {view_p.extent(0), view_p.extent(1)}), Kokkos::Experimental::WorkItemProperty::HintLightWeight );

    // // launch data-independent kernels
    // {
    //   auto sTimer = scoped_timer("kernel nodata");
    //   for (int i = 0; i < 10000; ++i){
    //     Kokkos::parallel_for("print some numbers",
    //       iterate_p_policy,
    //       KOKKOS_LAMBDA(int j, int k) {
    //         if((j*k)%1000 == 1){
    //           printf("%d,%d; ", j, k);
    //         }
    //       }
    //     );
    //   }
    // }

    // launch data-dependent kernels
    {
      auto sTimer = scoped_timer("kernel add");
      for (int i = 0; i < 10000; ++i){
        kernel_add(view_p, view_q, view_r, iterate_p_policy_1);
        kernel_add(view_p, view_r, view_q, iterate_p_policy_1);
        Kokkos::deep_copy(host_q, view_q);
        Kokkos::deep_copy(host_r, view_r);
      }
    }

    cuda1.fence();  
    printSomeContents(host_r);
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