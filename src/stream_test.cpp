#include "../include/cuda_helper.hpp"
#include "../include/stream_manager.hpp"
#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?

// Assert during Release builds for this file:
#undef NDEBUG
#include <cassert> // reinclude the header to update the definition of assert()

#include "stream_test.hpp"
int main(int argc, char *argv[]) {

  std::cout << "Starting priority pool manual test ..." << std::endl;
  test_pool_ref_counting<cuda_helper, priority_pool<cuda_helper>>(0, 2);
  std::cout << "Manual priority pool test successfull!" << std::endl;
  std::cout << std::endl;

  std::cout << "Starting priority pool wrapper objects test ..." << std::endl;
  test_pool_wrappers<cuda_helper, priority_pool<cuda_helper>>(0, 2);
  std::cout << "Wrapper object priority pool test successfull!" << std::endl;
  std::cout << std::endl;

  std::cout << "Starting multigpu priority pool wrapper objects test ..."
            << std::endl;
  test_pool_wrappers<
      cuda_helper,
      priority_pool_multi_gpu<cuda_helper, priority_pool<cuda_helper>>>(1, 2);
  std::cout << "Multigpu wrapper object priority pool test successfull!"
            << std::endl;
  std::cout << std::endl;

  //   // Round robin pool tests:
  std::cout << "Started manual round-robin pool test ..." << std::endl;
  test_pool_ref_counting<cuda_helper, round_robin_pool<cuda_helper>>(0, 2);
  std::cout << "Manual round-robin pool test successfull!" << std::endl;
  std::cout << std::endl;

  std::cout << "Starting round-robin pool wrapper objects test ..."
            << std::endl;
  test_pool_wrappers<cuda_helper, round_robin_pool<cuda_helper>>(0, 2);
  std::cout << "Wrapper object round-robin pool test successfull!" << std::endl;
  std::cout << std::endl;

  std::cout << "Starting multigpu round robin pool wrapper objects test... "
            << std::endl;
  test_pool_wrappers<
      cuda_helper,
      priority_pool_multi_gpu<cuda_helper, priority_pool<cuda_helper>>>(1, 2);
  std::cout << "Multigpu wrapper object round robin pool test successfull!"
            << std::endl;

  std::cout << "Starting memcpy tests... " << std::endl;
  test_pool_memcpy<cuda_helper, round_robin_pool<cuda_helper>>(0, 2);
  test_pool_memcpy<
      cuda_helper,
      multi_gpu_round_robin_pool<cuda_helper, round_robin_pool<cuda_helper>>>(
      1, 2);
  test_pool_memcpy<cuda_helper, priority_pool<cuda_helper>>(0, 2);
  test_pool_memcpy<
      cuda_helper,
      priority_pool_multi_gpu<cuda_helper, priority_pool<cuda_helper>>>(1, 2);

  // combo pool
  test_pool_memcpy<cuda_helper, multi_gpu_round_robin_pool<
                                    cuda_helper, priority_pool<cuda_helper>>>(
      1, 2);
  test_pool_memcpy<
      cuda_helper,
      priority_pool_multi_gpu<cuda_helper, round_robin_pool<cuda_helper>>>(1,
                                                                           2);
  std::cout << "Finished memcpy tests! " << std::endl;
}