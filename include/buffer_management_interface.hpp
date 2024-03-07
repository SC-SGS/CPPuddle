#ifndef BUFFER_MANAGEMENT_INTERFACE_HPP
#define BUFFER_MANAGEMENT_INTERFACE_HPP

#include "detail/buffer_management.hpp"

namespace cppuddle {
namespace memory_recycling {

/// Print performance counters of all buffer managers to stdout
inline void print_buffer_counters() {
  detail::buffer_interface::print_performance_counters();
}
/// Deletes all buffers (even ones still marked as used), delete the buffer 
/// managers and the recycler itself
inline void force_buffer_cleanup() { detail::buffer_interface::clean_all(); }

/// Deletes all buffers currently marked as unused
inline void unused_buffer_cleanup() {
  detail::buffer_interface::clean_unused_buffers();
}
/// Deletes all buffers (even ones still marked as used), delete the buffer 
/// managers and the recycler itself. Disallows further usage.
inline void finalize() { detail::buffer_interface::finalize(); }

} // namespace memory_recycling
} // end namespace cppuddle

#endif
