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
  Kokkos::parallel_for("print contents",
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
    printf("%d hpx threads \n ", hpx::get_os_thread_count());
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

    Kokkos::deep_copy(testView, deviceMirror);
    printf("ran minimal example \n ");
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
      if((j*k)%1000 == 1){
        printf("%d, %d, %d ; ", j, k, reduceView(j, k));
      }
      sum += reduceView(j, k);
    }, reductionResult);
// Kokkos::fence();
  return reductionResult;
}

Kokkos::View<double**, Kokkos::Cuda::array_layout, Kokkos::HostSpace> initializeAndMirrorToHost
      (Kokkos::View<double**, Kokkos::DefaultExecutionSpace>& testView){
  printf("%d hpx threads \n ", hpx::get_os_thread_count());
  Kokkos::parallel_for("init",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          {0, 0}, {testView.extent(0), testView.extent(1)}),
      KOKKOS_LAMBDA(int j, int k) {
          testView(j, k) = j*k;
      });
  Kokkos::fence();
  printSomeContents(testView);
  printf("mirror view \n ", hpx::get_os_thread_count());

  auto hostMirror = Kokkos::create_mirror_view(testView);

  Kokkos::deep_copy(testView, hostMirror);
	printf("ran other minimal example \n ");
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

    Kokkos::View<double**, Kokkos::HostSpace> testViewHost(
      Kokkos::ViewAllocateWithoutInitializing("test host"), 150, 560);
    auto deviceMirror = initializeAndMirrorToDevice(testViewHost);
    Kokkos::fence();
    // why the heck are all of these filled with zeros???
    printf ("\n reduce on device\n");
    printSomeContents(deviceMirror);
    auto result = getSumOfViewEntries(deviceMirror);
    Kokkos::fence();
    printf ("\n reduce on host\n");
    printSomeContents(testViewHost);
    auto result2 = getSumOfViewEntries(testViewHost);
    Kokkos::fence();

    printf("create default execution space view \n ");
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> testViewDevice(
		  Kokkos::ViewAllocateWithoutInitializing("state"), 150, 560);
    auto hostMirror = initializeAndMirrorToHost(testViewDevice);
    Kokkos::fence();
    // why the heck are all of these filled with zeros???
    printf ("\n devicetest\n");
    printSomeContents(testViewDevice);
    Kokkos::fence();
    printf ("\n hostmirror\n");
    printSomeContents(hostMirror);
    Kokkos::fence();

    // auto f = hpx::async([testViewHost]() { return minimalExample(testViewHost); });
    // // auto g = hpx::async([deviceMirror]() -> double { return getSumOfViewEntries(deviceMirror); });
    // auto g = hpx::async([]() -> void { return otherMinimalExample(); });

    // auto when = hpx::when_all(f, g);
    // when.wait();
    
    printf("Results: %lf %lf\n", result, result2);// g.get());

    Kokkos::fence();
  }
  Kokkos::finalize(); 
}

