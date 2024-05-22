// Copyright (c) 2023-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CPPUDDLE_CONFIG_HPP
#define CPPUDDLE_CONFIG_HPP


// Mutex configuration
//
#if defined(CPPUDDLE_HAVE_HPX) && defined(CPPUDDLE_HAVE_HPX_MUTEX)
#include <hpx/mutex.hpp>
#else
#include <mutex>
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

namespace cppuddle {

#if defined(CPPUDDLE_HAVE_HPX) && defined(CPPUDDLE_HAVE_HPX_MUTEX)
using mutex_t = hpx::spinlock_no_backoff;
#else
using mutex_t = std::mutex;
#endif

// Recycling configuration
// TODO Add warnings here

// Aggressive recycling configuration
// TODO Add warning here

// Aggregation Debug configuration
// TODO Add warning here

// Thread and MultiGPU configuration
//
constexpr size_t number_instances = CPPUDDLE_HAVE_NUMBER_BUCKETS;
static_assert(number_instances >= 1);
constexpr size_t max_number_gpus = CPPUDDLE_HAVE_MAX_NUMBER_GPUS;
#ifndef CPPUDDLE_HAVE_HPX
static_assert(max_number_gpus == 1, "Non HPX builds do not support multigpu");
#endif
static_assert(max_number_gpus > 0);

/// Uses HPX thread information to determine which GPU should be used
inline size_t get_device_id(const size_t number_gpus) {
#if defined(CPPUDDLE_HAVE_HPX) 
    assert(number_gpus <= max_number_gpus);
    return hpx::get_worker_thread_num() % number_gpus; 
#else
    return 0;
#endif
}

} // end namespace cppuddle

#endif
