#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include "../include/buffer_manager.hpp"
#include <cstdio>
#include <typeinfo>


// TODO Insert templated singleton submanager -> Eine Bufferliste pro Submanager. Ein Buffer is
// Do it is as subclass

// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{
  buffer_recycler::get<double>(1000);
  buffer_recycler::get<double>(1000);
  buffer_recycler::get<int>(1000);
  buffer_recycler::get<int>(1000);
  std::cout << std::endl;
  // Create Vectors with the new allocator

  buffer_recycler::clean();
}
