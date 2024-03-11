// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef STD_RECYCLING_ALLOCATORS_HPP
#define STD_RECYCLING_ALLOCATORS_HPP

#include "buffer_management_interface.hpp"

/// \file
/// Contains the recycling allocators (in the form of type aliases)
/// using the std memory allocator 

namespace cppuddle {
namespace memory_recycling {

namespace device_selection {
/// Dummy GPU selector. Needs to be defined for MultiGPU builds as the default /
/// select_device_functor does not compile for > 1 GPU (to make sure all /
/// relevant allocators support multigpu)
template <typename T> struct select_device_functor<T, std::allocator<T>> {
  void operator()(const size_t device_id) {}
};
} // namespace device_selection


/// Recycling allocator for std memory
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using recycle_std = detail::recycle_allocator<T, std::allocator<T>>;
/// Recycling allocator for boost aligned memory (reusing previous content as well)
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
using aggressive_recycle_std =
    detail::aggressive_recycle_allocator<T, std::allocator<T>>;

} // namespace memory_recycling
} // namespace cppuddle

#endif
