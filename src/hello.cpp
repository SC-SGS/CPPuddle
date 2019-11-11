#include <hpx/hpx_main.hpp>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

struct hello_world {
  KOKKOS_INLINE_FUNCTION
  void operator() (const int i) const {
    printf ("Hello from i = %i\n", i);
  }
};

int main (int argc, char* argv[]) {
  Kokkos::initialize (argc, argv);

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

  Kokkos::finalize (); 
}

