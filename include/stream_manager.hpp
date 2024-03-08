// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

#include "cppuddle/executor_recycling/executor_pools_interface.hpp"

template <typename Interface>
using round_robin_pool
    [[deprecated("Use cppuddle::executor_recycling::round_robin_pool_impl from "
                 "header executor_pools_management.hpp instead")]] =
        cppuddle::executor_recycling::round_robin_pool_impl<Interface>;

template <typename Interface>
using priority_pool
    [[deprecated("Use cppuddle::executor_recycling::priority_pool_impl from "
                 "header executor_pools_management.hpp instead")]] =
        cppuddle::executor_recycling::priority_pool_impl<Interface>;

using stream_pool
    [[deprecated("Use cppuddle::executor_recycling::executor_pool from "
                 "header executor_pools_management.hpp instead")]] =
        cppuddle::executor_recycling::executor_pool;

template <typename Interface, typename Pool>
using stream_interface
    [[deprecated("Use cppuddle::executor_recycling::executor_interface from "
                 "header executor_pools_management.hpp instead")]] =
        cppuddle::executor_recycling::executor_interface<Interface, Pool>;

#endif
