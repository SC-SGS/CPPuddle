// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define USE_HPX_MAIN
#include "../include/stream_manager.hpp"
#include <hpx/async_cuda/cuda_executor.hpp>
#ifdef USE_HPX_MAIN
#include <hpx/hpx_init.hpp>
#else
#include <hpx/hpx_main.hpp>
#endif
#include <hpx/include/async.hpp>

// Assert during Release builds as well for this file:
#undef NDEBUG
#include <cassert> // reinclude the header to update the definition of assert()

#include "stream_test.hpp"

#ifdef USE_HPX_MAIN
int hpx_main(int argc, char *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
  std::cout << "Starting ref counting tests ..." << std::endl;
  test_pool_ref_counting<hpx::cuda::experimental::cuda_executor,
                         priority_pool<hpx::cuda::experimental::cuda_executor>>(
      2, 0, false);
  test_pool_ref_counting<
      hpx::cuda::experimental::cuda_executor,
      round_robin_pool<hpx::cuda::experimental::cuda_executor>>(2, 0, false);
  test_pool_ref_counting<
      hpx::cuda::experimental::cuda_executor,
      multi_gpu_round_robin_pool<
          hpx::cuda::experimental::cuda_executor,
          round_robin_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1,
                                                                     false);
  test_pool_ref_counting<
      hpx::cuda::experimental::cuda_executor,
      priority_pool_multi_gpu<
          hpx::cuda::experimental::cuda_executor,
          priority_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1, false);
  test_pool_ref_counting<
      hpx::cuda::experimental::cuda_executor,
      multi_gpu_round_robin_pool<
          hpx::cuda::experimental::cuda_executor,
          priority_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1, false);
  test_pool_ref_counting<
      hpx::cuda::experimental::cuda_executor,
      priority_pool_multi_gpu<
          hpx::cuda::experimental::cuda_executor,
          round_robin_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1,
                                                                     false);
  std::cout << "Finished ref counting tests!" << std::endl;

  std::cout << "Starting wrapper objects tests ..." << std::endl;
  test_pool_wrappers<hpx::cuda::experimental::cuda_executor,
                     priority_pool<hpx::cuda::experimental::cuda_executor>>(
      2, 0, false);
  test_pool_wrappers<hpx::cuda::experimental::cuda_executor,
                     round_robin_pool<hpx::cuda::experimental::cuda_executor>>(
      2, 0, false);
  test_pool_wrappers<
      hpx::cuda::experimental::cuda_executor,
      multi_gpu_round_robin_pool<
          hpx::cuda::experimental::cuda_executor,
          round_robin_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1,
                                                                     false);
  test_pool_wrappers<
      hpx::cuda::experimental::cuda_executor,
      priority_pool_multi_gpu<
          hpx::cuda::experimental::cuda_executor,
          priority_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1, false);

  test_pool_wrappers<
      hpx::cuda::experimental::cuda_executor,
      multi_gpu_round_robin_pool<
          hpx::cuda::experimental::cuda_executor,
          priority_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1, false);
  test_pool_wrappers<
      hpx::cuda::experimental::cuda_executor,
      priority_pool_multi_gpu<
          hpx::cuda::experimental::cuda_executor,
          round_robin_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1,
                                                                     false);
  std::cout << "Finished wrapper objects tests!" << std::endl;

  std::cout << "Starting memcpy tests... " << std::endl;
  test_pool_memcpy<hpx::cuda::experimental::cuda_executor,
                   round_robin_pool<hpx::cuda::experimental::cuda_executor>>(
      2, 0, false);
  test_pool_memcpy<
      hpx::cuda::experimental::cuda_executor,
      multi_gpu_round_robin_pool<
          hpx::cuda::experimental::cuda_executor,
          round_robin_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1,
                                                                     false);
  test_pool_memcpy<hpx::cuda::experimental::cuda_executor,
                   priority_pool<hpx::cuda::experimental::cuda_executor>>(
      2, 0, false);
  test_pool_memcpy<hpx::cuda::experimental::cuda_executor,
                   priority_pool_multi_gpu<
                       hpx::cuda::experimental::cuda_executor,
                       priority_pool<hpx::cuda::experimental::cuda_executor>>>(
      2, 1, false);

  // combo pool
  test_pool_memcpy<hpx::cuda::experimental::cuda_executor,
                   multi_gpu_round_robin_pool<
                       hpx::cuda::experimental::cuda_executor,
                       priority_pool<hpx::cuda::experimental::cuda_executor>>>(
      2, 1, false);
  test_pool_memcpy<
      hpx::cuda::experimental::cuda_executor,
      priority_pool_multi_gpu<
          hpx::cuda::experimental::cuda_executor,
          round_robin_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1,
                                                                     false);
  std::cout << "Finished memcpy tests! " << std::endl;

  std::cout << "Starting memcpy polling tests... " << std::endl;
  {
    hpx::cuda::experimental::enable_user_polling polling_scope;
    test_pool_memcpy<hpx::cuda::experimental::cuda_executor,
                     round_robin_pool<hpx::cuda::experimental::cuda_executor>>(
        2, 0, true);
    test_pool_memcpy<
        hpx::cuda::experimental::cuda_executor,
        multi_gpu_round_robin_pool<
            hpx::cuda::experimental::cuda_executor,
            round_robin_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1,
                                                                       true);
    test_pool_memcpy<hpx::cuda::experimental::cuda_executor,
                     priority_pool<hpx::cuda::experimental::cuda_executor>>(
        2, 0, true);
    test_pool_memcpy<
        hpx::cuda::experimental::cuda_executor,
        priority_pool_multi_gpu<
            hpx::cuda::experimental::cuda_executor,
            priority_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1, true);

    // combo pool
    test_pool_memcpy<
        hpx::cuda::experimental::cuda_executor,
        multi_gpu_round_robin_pool<
            hpx::cuda::experimental::cuda_executor,
            priority_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1, true);
    test_pool_memcpy<
        hpx::cuda::experimental::cuda_executor,
        priority_pool_multi_gpu<
            hpx::cuda::experimental::cuda_executor,
            round_robin_pool<hpx::cuda::experimental::cuda_executor>>>(2, 1,
                                                                       true);
  }
  recycler::force_cleanup();
  std::cout << "Finished memcpy tests! " << std::endl;
  return hpx::finalize();
}

#ifdef USE_HPX_MAIN
int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
#endif
