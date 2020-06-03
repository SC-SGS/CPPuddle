#ifndef ALIGNED_BUFFER_UTIL_HPP
#define ALIGNED_BUFFER_UTIL_HPP

#include "buffer_manager.hpp"
#include <boost/align/aligned_allocator.hpp>

namespace recycler {
template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_aligned = detail::recycle_allocator<
    T, boost::alignment::aligned_allocator<T, alignement>>;
template <typename T, std::size_t alignement,
          std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_aligned = detail::aggressive_recycle_allocator<
    T, boost::alignment::aligned_allocator<T, alignement>>;
} // namespace recycler

#endif