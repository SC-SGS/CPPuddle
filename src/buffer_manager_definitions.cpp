#include "../include/buffer_manager.hpp"

// Instance defintions
std::unique_ptr<recycler::detail::buffer_recycler>
    recycler::detail::buffer_recycler::recycler_instance{};
std::mutex recycler::detail::buffer_recycler::mut{};