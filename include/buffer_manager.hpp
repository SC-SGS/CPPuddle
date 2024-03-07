#ifndef BUFFER_MANAGER_INTERFACE_HPP
#define BUFFER_MANAGER_HPP

#include "buffer_management_interface.hpp"
#include "std_recycling_allocators.hpp"

namespace recycler {

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_std
    [[deprecated("Use from header std_recycling_allocators.hpp instead")]] =
        cppuddle::recycle_std<T>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_aligned
    [[deprecated("Use from header std_recycling_allocators.hpp instead")]] =
        cppuddle::aggressive_recycle_std<T>;

[[deprecated("Use cppuddle::print_buffer_counters() instead")]] 
inline void print_performance_counters() {
  cppuddle::print_buffer_counters();
}
/// Deletes all buffers (even ones still marked as used), delete the buffer
/// managers and the recycler itself
[[deprecated("Use cppuddle::force_buffer_cleanup() instead")]]
inline void force_cleanup() { cppuddle::force_buffer_cleanup(); }
/// Deletes all buffers currently marked as unused
[[deprecated("Use cppuddle::unused_buffer_cleanup() instead")]]
inline void cleanup() { cppuddle::unused_buffer_cleanup(); }
/// Deletes all buffers (even ones still marked as used), delete the buffer
/// managers and the recycler itself. Disallows further usage.
[[deprecated("Use cppuddle::finalize() instead")]]
inline void finalize() { cppuddle::finalize(); }

} // end namespace cppuddle

#endif
