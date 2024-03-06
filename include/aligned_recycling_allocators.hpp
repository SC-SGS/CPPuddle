// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ALIGNED_RECYCLING_ALLOCATORS_HPP
#define ALIGNED_RECYCLING_ALLOCATORS_HPP

#include "detail/buffer_recycler.hpp"
#include <boost/align/aligned_allocator.hpp>

namespace cppuddle {
namespace device_selection {
template <typename T, size_t alignement>
struct select_device_functor<
    T, boost::alignment::aligned_allocator<T, alignement>> {
  void operator()(const size_t device_id) {}
};
} // namespace device_selection

template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_aligned = detail::recycle_allocator<
    T, boost::alignment::aligned_allocator<T, alignement>>;
template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_aligned = detail::aggressive_recycle_allocator<
    T, boost::alignment::aligned_allocator<T, alignement>>;
} // namespace cppuddle

#endif
