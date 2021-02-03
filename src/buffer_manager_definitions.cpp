// Copyright (c) 2020-2021 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "../include/buffer_manager.hpp"

// Instance defintions
std::unique_ptr<recycler::detail::buffer_recycler>
    recycler::detail::buffer_recycler::recycler_instance{};
std::mutex recycler::detail::buffer_recycler::mut{};
