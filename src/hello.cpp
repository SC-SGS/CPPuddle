
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

  printf ("Hello World on Kokkos execution space %s\n",
          typeid (Kokkos::DefaultExecutionSpace).name ());

  Kokkos::parallel_for ("HelloWorld",15, hello_world ());

  Kokkos::finalize ();
}

