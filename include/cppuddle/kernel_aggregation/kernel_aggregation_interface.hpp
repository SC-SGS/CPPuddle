// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef KERNEL_AGGREGATION_INTERFACE_HPP
#define KERNEL_AGGREGATION_INTERFACE_HPP

#include "cppuddle/executor_recycling/executor_pools_interface.hpp"
#include "cppuddle/kernel_aggregation/detail/aggregation_executors_and_allocators.hpp"
#include "cppuddle/kernel_aggregation/detail/aggregation_executor_pools.hpp"

/// \file
/// Kernel aggregation public interface

namespace cppuddle {
/// CPPuddle namespace containing the kernel aggregation functionality
namespace kernel_aggregation {

/// Possible launch modes:
/// EAGER   = launch either when enough kernels aggregated or executor becomes idles
/// STRICT  = launch only when enough kernels aggregated (be aware of deadlocks when not
///           enough kernels are available!)
/// ENDLESS = launch only when executor becomes idle
using aggregated_executor_modes =
    cppuddle::kernel_aggregation::detail::aggregated_executor_modes;

/// Allocator to get a buffer slice of a buffer shared with other
/// tasks in the same aggregation region
template <typename T, typename Host_Allocator, typename Executor>
using allocator_slice =
    cppuddle::kernel_aggregation::detail::allocator_slice<T, Host_Allocator, Executor>;

/// Executor facilitating the kernel aggregation
/// Contains the executor_slice subclass which is intended to be used
/// by the individual tasks
template <typename Executor>
using aggregated_executor =
    cppuddle::kernel_aggregation::detail::aggregated_executor<Executor>;

/// Pool to get an aggregation executor for the desired code region (kernelname)
template <const char *kernelname, class Interface, class Pool>
using aggregation_pool =
    cppuddle::kernel_aggregation::detail::aggregation_pool<kernelname, Interface,
    Pool>;

/// Start an aggregation region (passsed via lambda)
template <const char* region_name, typename executor_t, typename return_type>
hpx::future<return_type> aggregation_region(const size_t team_size,
    std::function<return_type(size_t, size_t,
        typename cppuddle::kernel_aggregation::detail::aggregated_executor<
            executor_t>::executor_slice&)> &&aggregation_area) {
    using aggregation_pool_t = cppuddle::kernel_aggregation::aggregation_pool<region_name,
        executor_t, cppuddle::executor_recycling::round_robin_pool_impl<executor_t>>;
    static hpx::once_flag pool_init;
    hpx::call_once(pool_init,
        detail::init_area_aggregation_pool<aggregation_pool_t>, team_size);
    auto executor_slice_fut = aggregation_pool_t::request_executor_slice();
    auto ret_fut = executor_slice_fut.value().then(hpx::annotated_function(
        [aggregation_area](auto &&fut) {
          typename cppuddle::kernel_aggregation::detail::aggregated_executor<
              executor_t>::Executor_Slice agg_exec = fut.get();
          const size_t slice_id = agg_exec.id;
          const size_t number_slices = agg_exec.number_slices;
          return aggregation_area(slice_id, number_slices, agg_exec);
        },
        region_name));
    return ret_fut;
}

} // namespace kernel_aggregation 
} // namespace cppuddle

#endif
