#include "../include/buffer_manager.hpp"
#include "../include/cuda_helper.hpp"
#include "../include/stream_manager.hpp"
#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?

int main(int argc, char *argv[]) {
  stream_pool::init<cuda_helper, round_robin_pool<cuda_helper>>(0, 32);

  hpx_stream_interface test(0);
  // gpu_stream_manager<hpx_stream_interface>(32, 0);

  {
    auto meins = stream_pool::get_interface<cuda_helper,
                                            round_robin_pool<cuda_helper>>();
    cuda_helper test = std::get<0>(meins);
  }
}