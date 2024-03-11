// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// DEPRECATED: Do not use this file
// Only intended to make the old interface work a bit longer.
// See deprecation warnings for the new location of the functionality

#ifndef ALIGNED_BUFFER_UTIL_HPP
#define ALIGNED_BUFFER_UTIL_HPP

#include "cppuddle/memory_recycling/aligned_recycling_allocators.hpp"

namespace recycler {

template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_aligned
    [[deprecated("Use from header aligned_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::recycle_aligned<T, alignement>;

template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_aligned
    [[deprecated("Use from header aligned_recycling_allocators.hpp instead")]] =
        cppuddle::memory_recycling::aggressive_recycle_aligned<T, alignement>;

} // namespace recycler

#endif
