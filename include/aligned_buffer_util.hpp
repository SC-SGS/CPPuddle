// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ALIGNED_BUFFER_UTIL_HPP
#define ALIGNED_BUFFER_UTIL_HPP

#include "aligned_recycling_allocators.hpp"

namespace recycler {

template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_aligned
    [[deprecated("Use from header aligned_recycling_allocators.hpp instead")]] =
        cppuddle::recycle_aligned<T, alignement>;

template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_aligned
    [[deprecated("Use from header aligned_recycling_allocators.hpp instead")]] =
        cppuddle::aggressive_recycle_aligned<T, alignement>;

} // namespace recycler

#endif
