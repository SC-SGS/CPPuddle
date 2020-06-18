#include "../include/stream_manager.hpp"

// Instance defintions
std::unique_ptr<stream_pool> stream_pool::access_instance{};
std::mutex stream_pool::mut{};
