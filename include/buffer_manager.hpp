// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// DEPRECATED: Do not use this file
// Only intended to make the old interface work a bit longer.
// See deprecation warnings for the new location of the functionality

#ifndef BUFFER_MANAGER_HPP
#define BUFFER_MANAGER_HPP

#include "cppuddle/common/config.hpp"
#include "cppuddle/memory_recycling/buffer_management_interface.hpp"
#include "cppuddle/memory_recycling/detail/buffer_management.hpp"
#include "cppuddle/memory_recycling/std_recycling_allocators.hpp"

/// Deprecated LEGACY namespace. Kept around for compatiblity with old code for now
namespace recycler {

namespace detail {
using buffer_recycler [[deprecated(
    "Use buffer_interface from header "
    "cppuddle/memory_recycling/detail/buffer_management.hpp instead")]] =
    cppuddle::memory_recycling::detail::buffer_interface;
}

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_std
    [[deprecated("Use from header std_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::recycle_std<T>;

template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_std
    [[deprecated("Use from header std_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::aggressive_recycle_std<T>;

[[deprecated("Use cppuddle::memory_recycling::print_buffer_counters() instead")]] 
inline void print_performance_counters() {
  cppuddle::memory_recycling::print_buffer_counters();
}
/// Deletes all buffers (even ones still marked as used), delete the buffer
/// managers and the recycler itself
[[deprecated("Use cppuddle::memory_recycling::force_buffer_cleanup() instead")]]
inline void force_cleanup() { cppuddle::memory_recycling::force_buffer_cleanup(); }
/// Deletes all buffers currently marked as unused
[[deprecated("Use cppuddle::memory_recycling::unused_buffer_cleanup() instead")]]
inline void cleanup() { cppuddle::memory_recycling::unused_buffer_cleanup(); }
/// Deletes all buffers (even ones still marked as used), delete the buffer
/// managers and the recycler itself. Disallows further usage.
[[deprecated("Use cppuddle::memory_recycling::finalize() instead")]]
inline void finalize() { cppuddle::memory_recycling::finalize(); }

[[deprecated("Use cppuddle::max_number_gpus instead")]] constexpr auto max_number_gpus =
    cppuddle::max_number_gpus;
[[deprecated("Use cppuddle::number_instances instead")]] constexpr auto number_instances =
    cppuddle::number_instances;

} // namespace recycler

#endif
