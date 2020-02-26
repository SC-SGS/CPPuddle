#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include "../include/buffer_manager.hpp"
#include <cstdio>
#include <typeinfo>


// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{
  // buffer_recycler::get<double>(1000);
  // buffer_recycler::get<double>(1000);
  // buffer_recycler::get<float>(1000);
  // buffer_recycler::get<float>(1000);
  // buffer_recycler::get<int>(1000);
  // buffer_recycler::get<int>(1000);
  // buffer_recycler::get<int>(1000);
  // buffer_recycler::get<int>(1000);
  // std::cout << std::endl;

  // buffer_recycler::clean_all();

  // Create Vectors with the new allocator
  {
    std::cout << "Creating first scope of vectors" << std::endl;
    std::vector<float, recycle_allocator<float>> test(200);
    std::vector<float, recycle_allocator<float>> test1(200);
    std::cout << "Leaving first scope ... " << std::endl;
  } // let Vectors run out of scope
  { // Check for memory reusage
    std::cout << "Creating second scope of vectors" << std::endl;
    std::vector<float, recycle_allocator<float>> test(2000);
    std::vector<float, recycle_allocator<float>> test1(200);
    std::vector<float, recycle_allocator<float>> test2(200);
    std::vector<double, recycle_allocator<double>> test3(200);
    std::cout << "Leaving second scope ... " << std::endl;
  }
}
