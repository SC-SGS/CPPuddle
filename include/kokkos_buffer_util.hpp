// Copyright (c) 2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// DEPRECATED: Do not use this file
// Only intended to make the old interface work a bit longer.
// See deprecation warnings for the new location of the functionality

#ifndef KOKKOS_BUFFER_UTIL_HPP
#define KOKKOS_BUFFER_UTIL_HPP
#include "cppuddle/memory_recycling/util/recycling_kokkos_view.hpp"


namespace recycler {
template <typename kokkos_type, typename alloc_type, typename element_type>
using aggregated_recycled_view [[deprecated(
    "Use aggregated_recycle_view from header recycling_kokkos_view.hpp "
    "instead")]] =
    cppuddle::memory_recycling::detail::aggregated_recycling_view<
        kokkos_type, alloc_type, element_type>;

template <typename kokkos_type, typename alloc_type, typename element_type>
using recycled_view [[deprecated(
    "Use recycle_view from header recycling_kokkos_view.hpp instead")]] =
    cppuddle::memory_recycling::recycling_view<kokkos_type, alloc_type, element_type>;

} // end namespace recycler

#endif
