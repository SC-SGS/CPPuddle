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

Kokkos::View<double**, Kokkos::Experimental::HPX::array_layout, Kokkos::Cuda::memory_space> 
  minimalExample(Kokkos::View<double**, Kokkos::HostSpace> testView){
    printf("%d hpx threads \n ", hpx::get_os_thread_count());
    Kokkos::parallel_for("init",
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {150, 560}),
        KOKKOS_LAMBDA(int j, int k) {
          if((j+k)%1000 == 0){
            printf("%d, %d ", j, k);
          }
			    volatile double a = std::pow(j*k, 3);
          testView(j, k) = a;
        });
	  Kokkos::fence();

    auto deviceMirror = Kokkos::create_mirror_view(typename Kokkos::DefaultExecutionSpace::memory_space(), testView);

    Kokkos::deep_copy(testView, deviceMirror);
    printf("ran minimal example \n ");
    Kokkos::fence();
    return deviceMirror;
}

double getSumOfViewEntries(Kokkos::View<double**, Kokkos::Experimental::HPX::array_layout,  
  Kokkos::Cuda::memory_space> deviceView){
      double reductionResult = 0;
      Kokkos::parallel_reduce("reduce all",
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
            {0, 0}, {deviceView.extent(0), deviceView.extent(1)}),
        KOKKOS_LAMBDA(int j, int k,  double& sum) {
          if((j*k)%1000 == 0){
            printf("%d, %d ", j, k);
          }
          sum += deviceView(j, k);
        }, reductionResult);
    // Kokkos::fence();
      return reductionResult;
}

void otherMinimalExample(){
    printf("create default execution space view \n ");
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> testView(
		Kokkos::ViewAllocateWithoutInitializing("state"), 150, 560);
    printf("%d hpx threads \n ", hpx::get_os_thread_count());
    Kokkos::parallel_for("init",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {150, 560}),
        KOKKOS_LAMBDA(int j, int k) {
            testView(j, k) = j*k;
        });
	Kokkos::fence();
    printf("mirror view \n ", hpx::get_os_thread_count());

    auto hostMirror = Kokkos::create_mirror_view(testView);

    Kokkos::deep_copy(testView, hostMirror);
	printf("ran other minimal example \n ");
	Kokkos::fence();
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


    Kokkos::View<double**, Kokkos::HostSpace> testViewHost(Kokkos::ViewAllocateWithoutInitializing("test host"), 150, 560);

    auto deviceMirror = minimalExample(testViewHost);
    Kokkos::fence();
    // auto result = getSumOfViewEntries(deviceMirror);
    otherMinimalExample();
    Kokkos::fence();

    auto f = hpx::async([testViewHost]() { return minimalExample(testViewHost); });
    // auto g = hpx::async([deviceMirror]() -> double { return getSumOfViewEntries(deviceMirror); });
    auto g = hpx::async([]() -> void { return otherMinimalExample(); });

    auto when = hpx::when_all(f, g);

    when.wait();
    // f.wait();
    // g.wait();
    // printf("Result: %lf\n", g.get());

    Kokkos::fence();
  }
  Kokkos::finalize(); 
}

