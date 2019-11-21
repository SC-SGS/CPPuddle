#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

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


int main (int argc, char* argv[]) {
  Kokkos::initialize (argc, argv);

  Kokkos::print_configuration(std::cout);
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

    // printSomeContents(testViewHost);
    // printSomeContents(deviceMirror);
    Kokkos::fence();

    // Mirror a device view to host
    printf("create default execution space view \n ");
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> testViewDevice(
		  Kokkos::ViewAllocateWithoutInitializing("state"), 150, 560);
    auto hostMirror = initializeAndMirrorToHost(testViewDevice);

    printf ("\n devicetest\n");
    printSomeContents(testViewDevice); 
    printf ("\n hostmirror\n");
    printSomeContents(hostMirror);
    Kokkos::fence();

                  
    // auto when = hpx::when_all(f, g);
    // terminate called after throwing an instance of 'hpx::detail::exception_with_info<hpx::exception>'
    //  what():  this future has no valid shared state: HPX(no_state)
    // when.wait();
    f.wait();
    g.wait();

    printf("Results: %lf %lf should be the same\n", g.get(), f.get());

    Kokkos::fence();
  }
  Kokkos::finalize(); 
}

