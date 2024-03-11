// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXECUTOR_POOLS_INTERFACE_HPP
#define EXECUTOR_POOLS_INTERFACE_HPP

#include "cppuddle/executor_recycling/detail/executor_pools_management.hpp"

namespace cppuddle {
namespace executor_recycling {

template <typename Interface>
using round_robin_pool_impl =
        detail::round_robin_pool_impl<Interface>;

template <typename Interface>
using priority_pool_impl =
        detail::priority_pool_impl<Interface>;

using executor_pool =
        detail::executor_pool;

template <typename Interface, typename Pool>
using executor_interface =
        detail::executor_interface<Interface, Pool>;

} // end namespace executor_recycling
} // end namespace cppuddle

#endif
