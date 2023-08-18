// Copyright (c) 2023-2023 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CPPUDDLE_CONFIG_HPP
#define CPPUDDLE_CONFIG_HPP

// Mutex configuration
//
#if defined(CPPUDDLE_HAVE_HPX) && defined(CPPUDDLE_HAVE_HPX_MUTEX)
#include <hpx/mutex.hpp>
using mutex_t = hpx::spinlock_no_backoff;
using aggregation_mutex_t = hpx::mutex;
#else
#include <mutex>
using mutex_t = std::mutex;
using aggregation_mutex_t = std::mutex;
#endif

// HPX-aware configuration
//
#ifdef CPPUDDLE_HAVE_HPX
#ifndef CPPUDDLE_HAVE_HPX_AWARE_ALLOCATORS
#pragma message                                                                \
"Warning: CPPuddle build with HPX support but without HPX-aware allocators enabled. \
For better performance configure CPPuddle with CPPUDDLE_WITH_HPX_AWARE_ALLOCATORS=ON!"
#else
// include runtime to get HPX thread IDs required for the HPX-aware allocators
#include <hpx/include/runtime.hpp>
#endif
#endif

// Recycling configuration
// TODO Add warnings here

// Aggressive recycling configuration
// TODO Add warning here

// Aggregation Debug configuration
// TODO Add warning here

// Thread and MultiGPU configuration
//
constexpr size_t number_instances = CPPUDDLE_MAX_NUMBER_WORKERS;
static_assert(number_instances >= 1);
constexpr size_t max_number_gpus = CPPUDDLE_MAX_NUMBER_GPUS;
#ifndef CPPUDDLE_HAVE_HPX
static_assert(max_number_gpus == 1, "Non HPX builds do not support multigpu");
#endif
static_assert(number_instances >= max_number_gpus);
static_assert(max_number_gpus > 0);
static_assert(number_instances > 0);
//constexpr size_t instances_per_gpu = number_instances / max_number_gpus;

/// Uses HPX thread information to determine which GPU should be used
inline size_t get_device_id(void) {
#if defined(CPPUDDLE_HAVE_HPX) 
    //return hpx::get_worker_thread_num() / max_num_gpus; 
    return 0; 
#else
    return 0;
#endif
}

#endif
