// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXECUTOR_POOLS_INTERFACE_HPP
#define EXECUTOR_POOLS_INTERFACE_HPP

/// \file
/// Executor recycling public interface

#include "cppuddle/executor_recycling/detail/executor_pools_management.hpp"

/// Primary CPPuddle namespace containing the three primary feature modules /
/// (memory_recycling, / executor_recycling and kernel_aggregation) in
/// sub-namespaces
namespace cppuddle {
/// CPPuddle namespace containing the executor pool functionality
namespace executor_recycling {

/// Round robin pool strategy implementation
template <typename Interface>
using round_robin_pool_impl =
        detail::round_robin_pool_impl<Interface>;

/// Priority pool strategy implementation
template <typename Interface>
using priority_pool_impl =
        detail::priority_pool_impl<Interface>;

/// Main access to all executor pools
using executor_pool =
        detail::executor_pool;

/// RAII wrapper for executors
template <typename Interface, typename Pool>
using executor_interface =
        detail::executor_interface<Interface, Pool>;

} // end namespace executor_recycling
} // end namespace cppuddle

#endif
