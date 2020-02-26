#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include "../include/buffer_manager.hpp"
#include <cstdio>
#include <typeinfo>


constexpr size_t number_futures = 64;
constexpr size_t array_size = 2000;
constexpr size_t passes = 5;

// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{

/** Stress test for concurrency and performance:
 *  Hopefully this will catch any race conditions and allow us to
 *  determine bottlelegs by evaluating the performance of different allocator 
 *  implementations.
 * */
  static_assert(passes >= 1);
  static_assert(array_size >= 1);
  assert(number_futures >= hpx::get_num_worker_threads());
  std::array<hpx::future<void>, number_futures> futs;
  for (size_t i = 0; i < number_futures; i++) {
    futs[i]=hpx::async([&]() {
      std::vector<float, recycle_allocator<float>> test0(array_size);
      std::vector<float, recycle_allocator<float>> test1(array_size);
      std::vector<float, recycle_allocator<float>> test2(array_size);
      std::vector<float, recycle_allocator<float>> test3(array_size);
      std::vector<double, recycle_allocator<double>> test4(array_size);
      std::vector<double, recycle_allocator<double>> test5(array_size);
      std::vector<double, recycle_allocator<double>> test6(array_size);
      std::vector<double, recycle_allocator<double>> test7(array_size);
    });
  }
  for (size_t pass = 1; pass < passes; pass++) {
    for (size_t i = 0; i < number_futures; i++) {
      futs[i] = futs[i].then([&](hpx::future<void> &&predecessor) {
        std::vector<float, recycle_allocator<float>> test0(array_size);
        std::vector<float, recycle_allocator<float>> test1(array_size);
        std::vector<float, recycle_allocator<float>> test2(array_size);
        std::vector<float, recycle_allocator<float>> test3(array_size);
        std::vector<double, recycle_allocator<double>> test4(array_size);
        std::vector<double, recycle_allocator<double>> test5(array_size);
        std::vector<double, recycle_allocator<double>> test6(array_size);
        std::vector<double, recycle_allocator<double>> test7(array_size);
      });
    }
  }
  auto when = hpx::when_all(futs);
  when.wait();
}
