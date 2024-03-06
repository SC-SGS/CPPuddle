#ifndef BUFFER_MANAGER_INTERFACE_HPP
#define BUFFER_MANAGER_HPP

#include "buffer_management_interface.hpp"

namespace recycler {

[[deprecated("Use cppuddle::print_buffer_counters() instead")]]
inline void print_performance_counters() { cppuddle::print_buffer_counters(); }
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
inline void finalize() { detail::buffer_interface::finalize(); }

} // end namespace cppuddle

#endif
