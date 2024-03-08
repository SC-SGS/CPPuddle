// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef AGGREGATION_MANAGER_HPP
#define AGGREGATION_MANAGER_HPP

#include "cppuddle/kernel_aggregation/kernel_aggregation_management.hpp"

using Aggregated_Executor_Modes =
    cppuddle::kernel_aggregation::aggregated_executor_modes;

template <typename T, typename Host_Allocator, typename Executor>
using Allocator_Slice =
    cppuddle::kernel_aggregation::allocator_slice<T, Host_Allocator, Executor>;

template <typename Executor>
using Aggregated_Executor =
    cppuddle::kernel_aggregation::Aggregated_Executor<Executor>;

template <const char *kernelname, class Interface, class Pool>
using aggregation_pool =
    cppuddle::kernel_aggregation::aggregation_pool<kernelname, Interface,
    Pool>;

#endif
