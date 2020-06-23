#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"
#include "../include/cuda_helper.hpp"
#include "../include/stream_manager.hpp"
#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?

// Assert during Release builds for this file:
#undef NDEBUG
#include <cassert> // reinclude the header to update the definition of assert()

int main(int argc, char *argv[]) {

  //   std::cout << "Starting priority pool manual test ..." << std::endl;
  //   stream_pool::init<cuda_helper, priority_pool<cuda_helper>>(0, 2);
  //   {
  //     auto test1 =
  //         stream_pool::get_interface<cuda_helper,
  //         priority_pool<cuda_helper>>();
  //     auto load1 = stream_pool::get_current_load<cuda_helper,
  //                                                priority_pool<cuda_helper>>();
  //     assert(load1 == 0);
  //     cuda_helper test1_interface = std::get<0>(test1);
  //     size_t test1_index = std::get<1>(test1);

  //     auto test2 =
  //         stream_pool::get_interface<cuda_helper,
  //         priority_pool<cuda_helper>>();
  //     auto load2 = stream_pool::get_current_load<cuda_helper,
  //                                                priority_pool<cuda_helper>>();
  //     assert(load2 == 1);
  //     cuda_helper test2_interface = std::get<0>(test2);
  //     auto fut = test2_interface.get_future();
  //     size_t test2_index = std::get<1>(test2);

  //     auto test3 =
  //         stream_pool::get_interface<cuda_helper,
  //         priority_pool<cuda_helper>>();
  //     auto load3 = stream_pool::get_current_load<cuda_helper,
  //                                                priority_pool<cuda_helper>>();
  //     assert(load3 == 1);
  //     cuda_helper test3_interface = std::get<0>(test3);
  //     size_t test3_index = std::get<1>(test3);

  //     auto test4 =
  //         stream_pool::get_interface<cuda_helper,
  //         priority_pool<cuda_helper>>();
  //     auto load4 = stream_pool::get_current_load<cuda_helper,
  //                                                priority_pool<cuda_helper>>();
  //     cuda_helper test4_interface = std::get<0>(test4);
  //     size_t test4_index = std::get<1>(test4);
  //     assert(load4 == 2);

  //     stream_pool::release_interface<cuda_helper,
  //     priority_pool<cuda_helper>>(
  //         test4_index);
  //     load4 = stream_pool::get_current_load<cuda_helper,
  //                                           priority_pool<cuda_helper>>();
  //     assert(load4 == 1);

  //     stream_pool::release_interface<cuda_helper,
  //     priority_pool<cuda_helper>>(
  //         test3_index);
  //     load3 = stream_pool::get_current_load<cuda_helper,
  //                                           priority_pool<cuda_helper>>();
  //     assert(load3 == 1);

  //     stream_pool::release_interface<cuda_helper,
  //     priority_pool<cuda_helper>>(
  //         test2_index);
  //     load2 = stream_pool::get_current_load<cuda_helper,
  //                                           priority_pool<cuda_helper>>();
  //     assert(load2 == 0);

  //     stream_pool::release_interface<cuda_helper,
  //     priority_pool<cuda_helper>>(
  //         test1_index);
  //     load1 = stream_pool::get_current_load<cuda_helper,
  //                                           priority_pool<cuda_helper>>();
  //     assert(load1 == 0);
  //   }
  //   auto load0 =
  //       stream_pool::get_current_load<cuda_helper,
  //       priority_pool<cuda_helper>>();
  //   assert(load0 == 0);
  //   std::cout << "Manual priority pool test successfull!" << std::endl;
  //   std::cout << std::endl;

  //   std::cout << "Starting priority pool wrapper objects test ..." <<
  //   std::endl;
  //   {
  //     hpx_stream_interface_pq test1(0);
  //     auto load = stream_pool::get_current_load<cuda_helper,
  //                                               priority_pool<cuda_helper>>();
  //     assert(load == 0);
  //     hpx_stream_interface_pq test2(0);
  //     load = stream_pool::get_current_load<cuda_helper,
  //                                          priority_pool<cuda_helper>>();
  //     auto fut = test2.get_future();
  //     assert(load == 1);
  //     hpx_stream_interface_pq test3(0);
  //     load = stream_pool::get_current_load<cuda_helper,
  //                                          priority_pool<cuda_helper>>();
  //     assert(load == 1);
  //     hpx_stream_interface_pq test4(0);
  //     load = stream_pool::get_current_load<cuda_helper,
  //                                          priority_pool<cuda_helper>>();
  //     assert(load == 2);

  //     // Check availability method:
  //     bool avail =
  //         stream_pool::interface_available<cuda_helper,
  //                                          priority_pool<cuda_helper>>(1);
  //     assert(avail == false); // NOLINT
  //     avail = stream_pool::interface_available<cuda_helper,
  //                                              priority_pool<cuda_helper>>(2);
  //     assert(avail == false); // NOLINT
  //     avail = stream_pool::interface_available<cuda_helper,
  //                                              priority_pool<cuda_helper>>(3);
  //     assert(avail == true); // NOLINT
  //   }
  //   load0 =
  //       stream_pool::get_current_load<cuda_helper,
  //       priority_pool<cuda_helper>>();
  //   assert(load0 == 0);
  //   std::cout << "Wrapper object priority pool test successfull!" <<
  //   std::endl; std::cout << std::endl;

  //   std::cout << "Starting multigpu priority pool wrapper objects test ..."
  //             << std::endl;
  //   stream_pool::init<
  //       cuda_helper,
  //       priority_pool_multi_gpu<cuda_helper, priority_pool<cuda_helper>>>(1,
  //       2);
  //   {
  //     hpx_stream_interface_mgpq test1(0);
  //     auto load = stream_pool::get_current_load<
  //         cuda_helper,
  //         priority_pool_multi_gpu<cuda_helper,
  //         priority_pool<cuda_helper>>>();
  //     assert(load == 0);
  //     hpx_stream_interface_mgpq test2(0);
  //     load = stream_pool::get_current_load<
  //         cuda_helper,
  //         priority_pool_multi_gpu<cuda_helper,
  //         priority_pool<cuda_helper>>>();
  //     assert(load == 1);
  //     hpx_stream_interface_mgpq test3(0);
  //     load = stream_pool::get_current_load<
  //         cuda_helper,
  //         priority_pool_multi_gpu<cuda_helper,
  //         priority_pool<cuda_helper>>>();
  //     assert(load == 1);
  //     hpx_stream_interface_mgpq test4(0);
  //     load = stream_pool::get_current_load<
  //         cuda_helper,
  //         priority_pool_multi_gpu<cuda_helper,
  //         priority_pool<cuda_helper>>>();
  //     assert(load == 2);

  //     // Check availability method:
  //     bool avail = stream_pool::interface_available<
  //         cuda_helper,
  //         priority_pool_multi_gpu<cuda_helper,
  //         priority_pool<cuda_helper>>>(1);
  //     assert(avail == false); // NOLINT
  //     avail = stream_pool::interface_available<
  //         cuda_helper,
  //         priority_pool_multi_gpu<cuda_helper,
  //         priority_pool<cuda_helper>>>(2);
  //     assert(avail == false); // NOLINT
  //     avail = stream_pool::interface_available<
  //         cuda_helper,
  //         priority_pool_multi_gpu<cuda_helper,
  //         priority_pool<cuda_helper>>>(3);
  //     assert(avail == true); // NOLINT
  //   }
  //   load0 = stream_pool::get_current_load<
  //       cuda_helper,
  //       priority_pool_multi_gpu<cuda_helper, priority_pool<cuda_helper>>>();
  //   assert(load0 == 0);
  //   std::cout << "Multigpu wrapper object priority pool test successfull!"
  //             << std::endl;
  //   std::cout << std::endl;

  //   // Round robin pool tests:
  //   std::cout << "Started manual round-robin pool test ..." << std::endl;
  //   stream_pool::init<cuda_helper, round_robin_pool<cuda_helper>>(0, 2);
  //   {
  //     auto test1 = stream_pool::get_interface<cuda_helper,
  //                                             round_robin_pool<cuda_helper>>();
  //     auto load1 = stream_pool::get_current_load<cuda_helper,
  //                                                round_robin_pool<cuda_helper>>();
  //     assert(load1 == 0);
  //     cuda_helper test1_interface = std::get<0>(test1);
  //     size_t test1_index = std::get<1>(test1);

  //     auto test2 = stream_pool::get_interface<cuda_helper,
  //                                             round_robin_pool<cuda_helper>>();
  //     auto load2 = stream_pool::get_current_load<cuda_helper,
  //                                                round_robin_pool<cuda_helper>>();
  //     assert(load2 == 1);
  //     cuda_helper test2_interface = std::get<0>(test1);
  //     size_t test2_index = std::get<1>(test2);

  //     auto test3 = stream_pool::get_interface<cuda_helper,
  //                                             round_robin_pool<cuda_helper>>();
  //     auto load3 = stream_pool::get_current_load<cuda_helper,
  //                                                round_robin_pool<cuda_helper>>();
  //     assert(load3 == 1);
  //     cuda_helper test3_interface = std::get<0>(test1);
  //     size_t test3_index = std::get<1>(test3);

  //     auto test4 = stream_pool::get_interface<cuda_helper,
  //                                             round_robin_pool<cuda_helper>>();
  //     auto load4 = stream_pool::get_current_load<cuda_helper,
  //                                                round_robin_pool<cuda_helper>>();
  //     assert(load4 == 2);
  //     cuda_helper test4_interface = std::get<0>(test1);
  //     size_t test4_index = std::get<1>(test4);

  //     stream_pool::release_interface<cuda_helper,
  //     round_robin_pool<cuda_helper>>(
  //         test4_index);
  //     load4 = stream_pool::get_current_load<cuda_helper,
  //                                           round_robin_pool<cuda_helper>>();
  //     assert(load4 == 1);

  //     stream_pool::release_interface<cuda_helper,
  //     round_robin_pool<cuda_helper>>(
  //         test3_index);
  //     load3 = stream_pool::get_current_load<cuda_helper,
  //                                           round_robin_pool<cuda_helper>>();
  //     assert(load3 == 1);

  //     stream_pool::release_interface<cuda_helper,
  //     round_robin_pool<cuda_helper>>(
  //         test2_index);
  //     load2 = stream_pool::get_current_load<cuda_helper,
  //                                           round_robin_pool<cuda_helper>>();
  //     assert(load2 == 0);

  //     stream_pool::release_interface<cuda_helper,
  //     round_robin_pool<cuda_helper>>(
  //         test1_index);
  //     load1 = stream_pool::get_current_load<cuda_helper,
  //                                           round_robin_pool<cuda_helper>>();
  //     assert(load1 == 0);
  //   }
  //   load0 = stream_pool::get_current_load<cuda_helper,
  //                                         round_robin_pool<cuda_helper>>();
  //   assert(load0 == 0);
  //   std::cout << "Manual round-robin pool test successfull!" << std::endl;
  //   std::cout << std::endl;

  //   std::cout << "Starting round-robin pool wrapper objects test ..."
  //             << std::endl;
  //   {
  //     hpx_stream_interface_rr test1(0);
  //     auto load = stream_pool::get_current_load<cuda_helper,
  //                                               round_robin_pool<cuda_helper>>();
  //     assert(load == 0);
  //     hpx_stream_interface_rr test2(0);
  //     load = stream_pool::get_current_load<cuda_helper,
  //                                          round_robin_pool<cuda_helper>>();
  //     assert(load == 1);
  //     hpx_stream_interface_rr test3(0);
  //     load = stream_pool::get_current_load<cuda_helper,
  //                                          round_robin_pool<cuda_helper>>();
  //     assert(load == 1);
  //     hpx_stream_interface_rr test4(0);
  //     load = stream_pool::get_current_load<cuda_helper,
  //                                          round_robin_pool<cuda_helper>>();
  //     assert(load == 2);

  //     // Check availability method:
  //     bool avail =
  //         stream_pool::interface_available<cuda_helper,
  //                                          round_robin_pool<cuda_helper>>(1);
  //     assert(avail == false); // NOLINT
  //     avail = stream_pool::interface_available<cuda_helper,
  //                                              round_robin_pool<cuda_helper>>(2);
  //     assert(avail == false); // NOLINT
  //     avail = stream_pool::interface_available<cuda_helper,
  //                                              round_robin_pool<cuda_helper>>(3);
  //     assert(avail == true); // NOLINT
  //   }
  //   load0 = stream_pool::get_current_load<cuda_helper,
  //                                         round_robin_pool<cuda_helper>>();
  //   assert(load0 == 0);
  //   std::cout << "Wrapper object round-robin pool test successfull!" <<
  //   std::endl; std::cout << std::endl;

  //   std::cout << "Starting multigpu round robin pool wrapper objects test
  //   ..."
  //             << std::endl;
  //   stream_pool::init<
  //       cuda_helper,
  //       multi_gpu_round_robin_pool<cuda_helper,
  //       round_robin_pool<cuda_helper>>>( 1, 2);
  //   {
  //     hpx_stream_interface_mgrr test1(0);
  //     auto load = stream_pool::get_current_load<
  //         cuda_helper, multi_gpu_round_robin_pool<
  //                          cuda_helper, round_robin_pool<cuda_helper>>>();
  //     assert(load == 0);
  //     hpx_stream_interface_mgrr test2(0);
  //     load = stream_pool::get_current_load<
  //         cuda_helper, multi_gpu_round_robin_pool<
  //                          cuda_helper, round_robin_pool<cuda_helper>>>();
  //     assert(load == 1);
  //     hpx_stream_interface_mgrr test3(0);
  //     load = stream_pool::get_current_load<
  //         cuda_helper, multi_gpu_round_robin_pool<
  //                          cuda_helper, round_robin_pool<cuda_helper>>>();
  //     assert(load == 1);
  //     hpx_stream_interface_mgrr test4(0);
  //     load = stream_pool::get_current_load<
  //         cuda_helper, multi_gpu_round_robin_pool<
  //                          cuda_helper, round_robin_pool<cuda_helper>>>();
  //     assert(load == 2);

  //     // Check availability method:
  //     bool avail = stream_pool::interface_available<
  //         cuda_helper,
  //         multi_gpu_round_robin_pool<cuda_helper,
  //         round_robin_pool<cuda_helper>>>( 1);
  //     assert(avail == false); // NOLINT
  //     avail = stream_pool::interface_available<
  //         cuda_helper,
  //         multi_gpu_round_robin_pool<cuda_helper,
  //         round_robin_pool<cuda_helper>>>( 2);
  //     assert(avail == false); // NOLINT
  //     avail = stream_pool::interface_available<
  //         cuda_helper,
  //         multi_gpu_round_robin_pool<cuda_helper,
  //         round_robin_pool<cuda_helper>>>( 3);
  //     assert(avail == true); // NOLINT
  //   }
  //   load0 = stream_pool::get_current_load<
  //       cuda_helper,
  //       multi_gpu_round_robin_pool<cuda_helper,
  //       round_robin_pool<cuda_helper>>>();
  //   assert(load0 == 0);
  //   std::cout << "Multigpu wrapper object round robin pool test successfull!"
  //             << std::endl;
  //   std::cout << std::endl;

  std::vector<double, recycler::recycle_allocator_cuda_host<double>> hostbuffer(
      512);
  recycler::cuda_device_buffer<double> devicebuffer(512);
  stream_pool::init<cuda_helper, round_robin_pool<cuda_helper>>(0, 2);
  std::cout << stream_pool::get_current_load<cuda_helper,
                                             round_robin_pool<cuda_helper>>()
            << std::endl;
  {
    auto test1 = stream_pool::get_interface<cuda_helper,
                                            round_robin_pool<cuda_helper>>();
    cuda_helper test1_interface = std::get<0>(test1);
    test1_interface.copy_async(devicebuffer.device_side_buffer,
                               hostbuffer.data(), 512 * sizeof(double),
                               cudaMemcpyHostToDevice);
    auto fut1 = test1_interface.get_future();
    fut1.get();
  }
}