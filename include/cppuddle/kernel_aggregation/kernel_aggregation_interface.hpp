// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef KERNEL_AGGREGATION_INTERFACE_HPP
#define KERNEL_AGGREGATION_INTERFACE_HPP

#include "cppuddle/kernel_aggregation/detail/aggregation_executors_and_allocators.hpp"
#include "cppuddle/kernel_aggregation/detail/aggregation_executor_pools.hpp"

namespace cppuddle {
namespace kernel_aggregation {

using aggregated_executor_modes =
    cppuddle::kernel_aggregation::detail::aggregated_executor_modes;

template <typename T, typename Host_Allocator, typename Executor>
using allocator_slice =
    cppuddle::kernel_aggregation::detail::allocator_slice<T, Host_Allocator, Executor>;

template <typename Executor>
using aggregated_executor =
    cppuddle::kernel_aggregation::detail::aggregated_executor<Executor>;

template <const char *kernelname, class Interface, class Pool>
using aggregation_pool =
    cppuddle::kernel_aggregation::detail::aggregation_pool<kernelname, Interface,
    Pool>;

} // namespace kernel_aggregation 
} // namespace cppuddle

#endif
