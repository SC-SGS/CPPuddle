// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ALIGNED_BUFFER_UTIL_HPP
#define ALIGNED_BUFFER_UTIL_HPP

#include "buffer_manager.hpp"
#include <boost/align/aligned_allocator.hpp>
#ifdef CPPUDDLE_HAVE_HPX
#include "hpx_buffer_util.hpp"
#endif

namespace recycler {
template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_aligned = detail::recycle_allocator<
    T, boost::alignment::aligned_allocator<T, alignement>>;
template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_aligned = detail::aggressive_recycle_allocator<
    T, boost::alignment::aligned_allocator<T, alignement>>;
#ifdef CPPUDDLE_HAVE_HPX
template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using numa_aware_recycle_aligned = detail::numa_aware_recycle_allocator<
    T, boost::alignment::aligned_allocator<T, alignement>>;
template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using numa_aware_aggressive_recycle_aligned =
    detail::numa_aware_aggressive_recycle_allocator<
        T, boost::alignment::aligned_allocator<T, alignement>>;
#endif
} // namespace recycler

#endif
