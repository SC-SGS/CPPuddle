
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

  printf ("Hello World on Kokkos (default) execution space %s\n",
          typeid (Kokkos::DefaultExecutionSpace).name ());

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

