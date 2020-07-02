#include "../include/stream_manager.hpp"
#include <hpx/async_cuda/cuda_executor.hpp>
#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?

// Assert during Release builds as well for this file:
#undef NDEBUG
#include <cassert> // reinclude the header to update the definition of assert()

#include "stream_test.hpp"
int main(int argc, char *argv[]) {

  std::cout << "Starting ref counting tests ..." << std::endl;
  test_pool_ref_counting<hpx::cuda::cuda_executor,
                         priority_pool<hpx::cuda::cuda_executor>>(2, 0, false);
  test_pool_ref_counting<hpx::cuda::cuda_executor,
                         round_robin_pool<hpx::cuda::cuda_executor>>(2, 0,
                                                                     false);
  test_pool_ref_counting<
      hpx::cuda::cuda_executor,
      multi_gpu_round_robin_pool<hpx::cuda::cuda_executor,
                                 round_robin_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  test_pool_ref_counting<
      hpx::cuda::cuda_executor,
      priority_pool_multi_gpu<hpx::cuda::cuda_executor,
                              priority_pool<hpx::cuda::cuda_executor>>>(2, 1,
                                                                        false);
  test_pool_ref_counting<
      hpx::cuda::cuda_executor,
      multi_gpu_round_robin_pool<hpx::cuda::cuda_executor,
                                 priority_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  test_pool_ref_counting<
      hpx::cuda::cuda_executor,
      priority_pool_multi_gpu<hpx::cuda::cuda_executor,
                              round_robin_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  std::cout << "Finished ref counting tests!" << std::endl;

  std::cout << "Starting wrapper objects tests ..." << std::endl;
  test_pool_wrappers<hpx::cuda::cuda_executor,
                     priority_pool<hpx::cuda::cuda_executor>>(2, 0, false);
  test_pool_wrappers<hpx::cuda::cuda_executor,
                     round_robin_pool<hpx::cuda::cuda_executor>>(2, 0, false);
  test_pool_wrappers<
      hpx::cuda::cuda_executor,
      multi_gpu_round_robin_pool<hpx::cuda::cuda_executor,
                                 round_robin_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  test_pool_wrappers<
      hpx::cuda::cuda_executor,
      priority_pool_multi_gpu<hpx::cuda::cuda_executor,
                              priority_pool<hpx::cuda::cuda_executor>>>(2, 1,
                                                                        false);

  test_pool_wrappers<
      hpx::cuda::cuda_executor,
      multi_gpu_round_robin_pool<hpx::cuda::cuda_executor,
                                 priority_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  test_pool_wrappers<
      hpx::cuda::cuda_executor,
      priority_pool_multi_gpu<hpx::cuda::cuda_executor,
                              round_robin_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  std::cout << "Finished wrapper objects tests!" << std::endl;

  std::cout << "Starting memcpy tests... " << std::endl;
  test_pool_memcpy<hpx::cuda::cuda_executor,
                   round_robin_pool<hpx::cuda::cuda_executor>>(2, 0, false);
  test_pool_memcpy<
      hpx::cuda::cuda_executor,
      multi_gpu_round_robin_pool<hpx::cuda::cuda_executor,
                                 round_robin_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  test_pool_memcpy<hpx::cuda::cuda_executor,
                   priority_pool<hpx::cuda::cuda_executor>>(2, 0, false);
  test_pool_memcpy<
      hpx::cuda::cuda_executor,
      priority_pool_multi_gpu<hpx::cuda::cuda_executor,
                              priority_pool<hpx::cuda::cuda_executor>>>(2, 1,
                                                                        false);

  // combo pool
  test_pool_memcpy<
      hpx::cuda::cuda_executor,
      multi_gpu_round_robin_pool<hpx::cuda::cuda_executor,
                                 priority_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  test_pool_memcpy<
      hpx::cuda::cuda_executor,
      priority_pool_multi_gpu<hpx::cuda::cuda_executor,
                              round_robin_pool<hpx::cuda::cuda_executor>>>(
      2, 1, false);
  std::cout << "Finished memcpy tests! " << std::endl;

  std::cout << "Starting memcpy polling tests... " << std::endl;
  {
    hpx::cuda::enable_user_polling polling_scope;
    test_pool_memcpy<hpx::cuda::cuda_executor,
                     round_robin_pool<hpx::cuda::cuda_executor>>(2, 0, true);
    test_pool_memcpy<
        hpx::cuda::cuda_executor,
        multi_gpu_round_robin_pool<hpx::cuda::cuda_executor,
                                   round_robin_pool<hpx::cuda::cuda_executor>>>(
        2, 1, true);
    test_pool_memcpy<hpx::cuda::cuda_executor,
                     priority_pool<hpx::cuda::cuda_executor>>(2, 0, true);
    test_pool_memcpy<
        hpx::cuda::cuda_executor,
        priority_pool_multi_gpu<hpx::cuda::cuda_executor,
                                priority_pool<hpx::cuda::cuda_executor>>>(2, 1,
                                                                          true);

    // combo pool
    test_pool_memcpy<
        hpx::cuda::cuda_executor,
        multi_gpu_round_robin_pool<hpx::cuda::cuda_executor,
                                   priority_pool<hpx::cuda::cuda_executor>>>(
        2, 1, true);
    test_pool_memcpy<
        hpx::cuda::cuda_executor,
        priority_pool_multi_gpu<hpx::cuda::cuda_executor,
                                round_robin_pool<hpx::cuda::cuda_executor>>>(
        2, 1, true);
  }
  std::cout << "Finished memcpy tests! " << std::endl;
}