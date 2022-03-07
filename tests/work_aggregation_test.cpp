// Copyright (c) 2022-2022 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#undef NDEBUG


#include "../include/aggregation_manager.hpp"
#include "../include/cuda_buffer_util.hpp"

#include <boost/program_options.hpp>


//===============================================================================
//===============================================================================
// Example functions

void print_stuff_error(int i) { hpx::cout << "i is not " << i << std::endl; }
void print_stuff1(int i) { hpx::cout << "i is " << i << std::endl; }
void print_stuff2(int i, double d) {
  hpx::cout << "i is " << i << std::endl;
  hpx::cout << "d is " << d << std::endl;
}
void print_stuff3(int i) { hpx::cout << "i is " << i << std::endl; }

size_t add_pointer_launches = 0.0;
template <typename T>
void add_pointer(size_t aggregation_size, T *A, T *B, T *C) {
  add_pointer_launches++;
  const size_t start_id = 0;
  for (size_t i = 0; i < aggregation_size; i++) {
    C[start_id + i] = B[start_id + i] + A[start_id + i];
  }
}

size_t add_launches = 0.0;
template <typename Container>
void add(size_t slice_size, Container &A, Container &B, Container &C) {
  add_launches++;
  const size_t start_id = 0;
  for (size_t i = 0; i < 4 * slice_size; i++) {
    C[start_id + i] = B[start_id + i] + A[start_id + i];
  }
}

//===============================================================================
//===============================================================================
//
/// Dummy CPU executor (providing correct interface but running everything
/// immediately Intended for testing the aggregation on the CPU, not for
/// production use!
struct Dummy_Executor {
  /// Executor is always ready
  hpx::lcos::future<void> get_future() {
    // To trigger interruption in exeuctor coalesing manually with the promise
    // For a proper CUDA executor we would get a future that's ready once the
    // stream is ready of course!
    return hpx::make_ready_future();
  }
  /// post -- executes immediately
  template <typename F, typename... Ts> void post(F &&f, Ts &&...ts) {
    f(std::forward<Ts>(ts)...);
  }
  /// async -- executores immediately and returns ready future
  template <typename F, typename... Ts>
  hpx::lcos::future<void> async(F &&f, Ts &&...ts) {
    f(std::forward<Ts>(ts)...);
    return hpx::make_ready_future();
  }
};

//===============================================================================
//===============================================================================
// Test scenarios
//

void sequential_test(void) {
  static const char kernelname[] = "kernel1";
  using kernel_pool1 = aggregation_pool<kernelname, Dummy_Executor,
                                        round_robin_pool<Dummy_Executor>>;
  kernel_pool1::init(8, 2, Aggregated_Executor_Modes::STRICT);
  // Sequential test
  hpx::cout << "Sequential test with all executor slices" << std::endl;
  hpx::cout << "----------------------------------------" << std::endl;
  {
    std::vector<hpx::lcos::future<void>> slices_done_futs;

    auto slice_fut1 = kernel_pool1::request_executor_slice();
    if (slice_fut1.has_value()) {
      slices_done_futs.emplace_back(slice_fut1.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        auto alloc_int =
            slice_exec.template make_allocator<int, std::allocator<int>>();
        hpx::cout << "Executor 1 ID is " << slice_exec.id << std::endl;
        std::vector<float, decltype(alloc)> some_data(
            slice_exec.number_slices * 10, float{}, alloc);
        std::vector<float, decltype(alloc)> some_data2(
            slice_exec.number_slices * 20, float{}, alloc);
        std::vector<int, decltype(alloc_int)> some_ints(
            slice_exec.number_slices * 20, int{}, alloc_int);
        std::vector<float, decltype(alloc)> some_vector(
            slice_exec.number_slices * 10, float{}, alloc);
        hpx::cout << "Executor 1 Data address is " << some_data.data()
                  << std::endl;

        int i = 1;
        float j = 2;
        slice_exec.post(print_stuff1, i);
        slice_exec.post(print_stuff2, i, j);
        auto kernel_fut = slice_exec.async(print_stuff1, i);
        kernel_fut.get();
      }));
    } else {
      hpx::cout << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut2 = kernel_pool1::request_executor_slice();
    if (slice_fut2.has_value()) {
      slices_done_futs.emplace_back(slice_fut2.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        auto alloc_int =
            slice_exec.template make_allocator<int, std::allocator<int>>();
        hpx::cout << "Executor 2 ID is " << slice_exec.id << std::endl;
        std::vector<float, decltype(alloc)> some_data(
            slice_exec.number_slices * 10, float{}, alloc);
        std::vector<float, decltype(alloc)> some_data2(
            slice_exec.number_slices * 20, float{}, alloc);
        std::vector<int, decltype(alloc_int)> some_ints(
            slice_exec.number_slices * 20, int{}, alloc_int);
        std::vector<float, decltype(alloc)> some_vector(
            slice_exec.number_slices * 10, float{}, alloc);
        hpx::cout << "Executor 2 Data address is " << some_data.data()
                  << std::endl;
        int i = 1;
        float j = 2;
        slice_exec.post(print_stuff1, i);
        slice_exec.post(print_stuff2, i, j);
        auto kernel_fut = slice_exec.async(print_stuff1, i);
        kernel_fut.get();
      }));
    } else {
      hpx::cout << "ERROR: Slice 2 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 2 was not created properly");
    }

    auto slice_fut3 = kernel_pool1::request_executor_slice();
    if (slice_fut3.has_value()) {
      slices_done_futs.emplace_back(slice_fut3.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        auto alloc_int =
            slice_exec.template make_allocator<int, std::allocator<int>>();
        hpx::cout << "Executor 3 ID is " << slice_exec.id << std::endl;
        std::vector<float, decltype(alloc)> some_data(
            slice_exec.number_slices * 10, float{}, alloc);
        std::vector<float, decltype(alloc)> somedata2(
            slice_exec.number_slices * 20, float{}, alloc);
        std::vector<int, decltype(alloc_int)> some_ints(
            slice_exec.number_slices * 20, int{}, alloc_int);
        std::vector<float, decltype(alloc)> some_vector(
            slice_exec.number_slices * 10, float{}, alloc);
        hpx::cout << "Executor 3 Data address is " << some_data.data()
                  << std::endl;
        int i = 1;
        float j = 2;
        slice_exec.post(print_stuff1, i);
        slice_exec.post(print_stuff2, i, j);
        auto kernel_fut = slice_exec.async(print_stuff1, i);
        kernel_fut.get();
      }));
    } else {
      hpx::cout << "ERROR: Slice 3 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 3 was not created properly");
    }

    auto slice_fut4 = kernel_pool1::request_executor_slice();
    if (slice_fut4.has_value()) {
      slices_done_futs.emplace_back(slice_fut4.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        auto alloc_int =
            slice_exec.template make_allocator<int, std::allocator<int>>();
        hpx::cout << "Executor 4 ID is " << slice_exec.id << std::endl;
        std::vector<float, decltype(alloc)> some_data(
            slice_exec.number_slices * 10, float{}, alloc);
        std::vector<float, decltype(alloc)> some_data2(
            slice_exec.number_slices * 20, float{}, alloc);
        std::vector<int, decltype(alloc_int)> some_ints(
            slice_exec.number_slices * 20, int{}, alloc_int);
        std::vector<float, decltype(alloc)> some_vector(
            slice_exec.number_slices * 10, float{}, alloc);
        hpx::cout << "Executor 4 Data address is " << some_data.data()
                  << std::endl;
        int i = 1;
        float j = 2;
        slice_exec.post(print_stuff1, i);
        slice_exec.post(print_stuff2, i, j);
        auto kernel_fut = slice_exec.async(print_stuff1, i);
        kernel_fut.get();
      }));
    } else {
      hpx::cout << "ERROR: Slice 4 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 4 was not created properly");
    }
    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by equesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
  }
  hpx::cout << std::endl;
}

void interruption_test(void) {
  // Interruption test
  hpx::cout << "Sequential test with interruption:" << std::endl;
  hpx::cout << "----------------------------------" << std::endl;
  {
    Aggregated_Executor<Dummy_Executor> agg_exec{
        4, Aggregated_Executor_Modes::EAGER};
    std::vector<hpx::lcos::future<void>> slices_done_futs;

    auto slice_fut1 = agg_exec.request_executor_slice();
    if (slice_fut1.has_value()) {
      slices_done_futs.emplace_back(slice_fut1.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        hpx::cout << "Got executor 1" << std::endl;
        slice_exec.post(print_stuff1, 1);
        slice_exec.post(print_stuff2, 1, 2.0);
        auto kernel_fut = slice_exec.async(print_stuff1, 1);
        kernel_fut.get();
      }));
    } else {
      hpx::cout << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    /*auto slice_fut2 = agg_exec.request_executor_slice();
    if (slice_fut2.has_value()) {
      slices_done_futs.emplace_back(slice_fut2.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        hpx::cout << "Got executor 2" << std::endl;
        slice_exec.post(print_stuff1, 1);
        slice_exec.post(print_stuff2, 1, 2.0);
        auto kernel_fut = slice_exec.async(print_stuff1, 1);
        kernel_fut.get();
      }));
    } else {
      hpx::cout << "ERROR: Slice 2 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 2 was not created properly");
    }

    auto slice_fut3 = agg_exec.request_executor_slice();
    if (slice_fut3.has_value()) {
      slices_done_futs.emplace_back(slice_fut3.value().then([](auto &&fut) {
        auto slice_exec = fut.get();
        hpx::cout << "Got executor 3" << std::endl;
        slice_exec.post(print_stuff1, 1);
        slice_exec.post(print_stuff2, 1, 2.0);
        auto kernel_fut = slice_exec.async(print_stuff1, 1);
        kernel_fut.get();
      }));
    } else {
      hpx::cout << "ERROR: Slice 3 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 3 was not created properly");
    }*/

    hpx::cout << "Requested 1 executors!" << std::endl;
    hpx::cout << "Realizing by setting the continuation future..." << std::endl;
    // Interrupt - should cause executor to start executing all slices
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
  }
  hpx::cout << std::endl;
  // recycler::force_cleanup();
}

void failure_test(void) {
  // Error test
  hpx::cout << "Error test with all wrong types and values in 2 slices"
            << std::endl;
  hpx::cout << "------------------------------------------------------"
            << std::endl;
  {
    Aggregated_Executor<Dummy_Executor> agg_exec{
        4, Aggregated_Executor_Modes::STRICT};

    auto slice_fut1 = agg_exec.request_executor_slice();

    std::vector<hpx::lcos::future<void>> slices_done_futs;
    if (slice_fut1.has_value()) {
    slices_done_futs.emplace_back(slice_fut1.value().then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 1" << std::endl;
      slice_exec.post(print_stuff1, 2);
     // auto async_fut = slice_exec.async(print_stuff1, 3);
     // async_fut.get();
    }));
    } else {
      hpx::cout << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut2 = agg_exec.request_executor_slice();
    if (slice_fut2.has_value()) {
    slices_done_futs.emplace_back(slice_fut2.value().then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 2" << std::endl;
      slice_exec.post(print_stuff1, 2);
     // auto async_fut = slice_exec.async(print_stuff1, 3);
     // async_fut.get();
    }));
    } else {
      hpx::cout << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut3 = agg_exec.request_executor_slice();
    if (slice_fut3.has_value()) {
    slices_done_futs.emplace_back(slice_fut3.value().then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 3" << std::endl;
      slice_exec.post(print_stuff_error, 2);
     // auto async_fut = slice_exec.async(print_stuff_error, 3);
      //async_fut.get();
    }));
    } else {
      hpx::cout << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut4 = agg_exec.request_executor_slice();
    if (slice_fut4.has_value()) {
    slices_done_futs.emplace_back(slice_fut4.value().then([](auto &&fut) {
      auto slice_exec = fut.get();
      hpx::cout << "Got executor 4" << std::endl;
      slice_exec.post(print_stuff1, 2.0f);
     // auto async_fut = slice_exec.async(print_stuff1, 3.0f);
     // async_fut.get();
    }));
    } else {
      hpx::cout << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by equesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
  }
  hpx::cout << std::endl;
}

void pointer_add_test(void) {
  hpx::cout << "Host aggregated add pointer example (no references used)"
            << std::endl;
  hpx::cout << "--------------------------------------------------------"
            << std::endl;
  static const char kernelname2[] = "kernel2";
  using kernel_pool2 = aggregation_pool<kernelname2, Dummy_Executor,
                                        round_robin_pool<Dummy_Executor>>;
  kernel_pool2::init(8, 2, Aggregated_Executor_Modes::STRICT);
  {
    std::vector<float> erg(512);
    std::vector<hpx::lcos::future<void>> slices_done_futs;

    auto slice_fut1 = kernel_pool2::request_executor_slice();

    if (slice_fut1.has_value()) {
      slices_done_futs.emplace_back(slice_fut1.value().then([&erg](auto &&fut) {
        // Get slice executor
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        size_t chunksize = 512 / slice_exec.number_slices;
        const size_t task_id = 0;
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(128 * slice_exec.number_slices,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = task_id + 1;
          B[i] = 2 * task_id;
        }

        // Run add function
        auto kernel_fut =
            slice_exec.async(add_pointer<float>, slice_exec.number_slices * 128,
                             A.data(), B.data(), C.data());
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = task_id * 128, j = slice_exec.id * 128;
             i < (task_id + 1) * 128; i++, j++) {
          erg[i] = C[j];
        }
      }));
    } else {
      hpx::cout << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut2 = kernel_pool2::request_executor_slice();
    if (slice_fut2.has_value()) {
      slices_done_futs.emplace_back(slice_fut2.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        size_t chunksize = 512 / slice_exec.number_slices;
        const size_t task_id = 1;
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(128 * slice_exec.number_slices,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = task_id + 1;
          B[i] = 2 * task_id;
        }

        // Run add function
        auto kernel_fut =
            slice_exec.async(add_pointer<float>, slice_exec.number_slices * 128,
                             A.data(), B.data(), C.data());
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = task_id * 128, j = slice_exec.id * 128;
             i < (task_id + 1) * 128; i++, j++) {
          erg[i] = C[j];
        }
      }));
    } else {
      hpx::cout << "ERROR: Slice 2 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 2 was not created properly");
    }

    auto slice_fut3 = kernel_pool2::request_executor_slice();
    if (slice_fut3.has_value()) {
      slices_done_futs.emplace_back(slice_fut3.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        size_t chunksize = 512 / slice_exec.number_slices;
        const size_t task_id = 2;
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(128 * slice_exec.number_slices,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = task_id + 1;
          B[i] = 2 * task_id;
        }

        // Run add function
        auto kernel_fut =
            slice_exec.async(add_pointer<float>, slice_exec.number_slices * 128,
                             A.data(), B.data(), C.data());
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = task_id * 128, j = slice_exec.id * 128;
             i < (task_id + 1) * 128; i++, j++) {
          erg[i] = C[j];
        }
      }));
    } else {
      hpx::cout << "ERROR: Slice 3 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 3 was not created properly");
    }

    auto slice_fut4 = kernel_pool2::request_executor_slice();
    if (slice_fut4.has_value()) {
      slices_done_futs.emplace_back(slice_fut4.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        size_t chunksize = 512 / slice_exec.number_slices;
        const size_t task_id = 3;
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(128 * slice_exec.number_slices,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(128 * slice_exec.number_slices,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = task_id + 1;
          B[i] = 2 * task_id;
        }

        // Run add function
        auto kernel_fut =
            slice_exec.async(add_pointer<float>, slice_exec.number_slices * 128,
                             A.data(), B.data(), C.data());
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = task_id * 128, j = slice_exec.id * 128;
             i < (task_id + 1) * 128; i++, j++) {
          erg[i] = C[j];
        }
      }));
    } else {
      hpx::cout << "ERROR: Slice 4 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 4 was not created properly");
    }
    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by requesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();

    hpx::cout << "Number add_pointer_launches=" << add_pointer_launches
              << std::endl;
    assert(add_pointer_launches == 2);
    hpx::cout << "Checking erg: " << std::endl;
    for (int slice = 0; slice < 4; slice++) {
      for (int i = slice * 128; i < (slice + 1) * 128; i++) {
        assert(erg[i] == 3 * slice + 1);
        hpx::cout << erg[i] << " ";
      }
    }
    hpx::cout << std::endl;
  }
  // recycler::force_cleanup();
  hpx::cout << std::endl;
}

void references_add_test(void) {
  hpx::cout << "Host aggregated add vector example (references used)"
            << std::endl;
  hpx::cout << "----------------------------------------------------"
            << std::endl;
  {
    /*Aggregated_Executor<decltype(executor1)> agg_exec{
        4, Aggregated_Executor_Modes::STRICT};*/
    auto &agg_exec =
        std::get<0>(stream_pool::get_interface<
                    Aggregated_Executor<Dummy_Executor>,
                    round_robin_pool<Aggregated_Executor<Dummy_Executor>>>());
    std::vector<float> erg(512);
    std::vector<hpx::lcos::future<void>> slices_done_futs;

    auto slice_fut1 = agg_exec.request_executor_slice();
    if (slice_fut1.has_value()) {
      slices_done_futs.emplace_back(slice_fut1.value().then([&erg](auto &&fut) {
        // Get slice executor
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(slice_exec.number_slices * 128,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = slice_exec.id + 1;
          B[i] = 2 * slice_exec.id;
        }

        // Run add function
        auto kernel_fut = slice_exec.async(add<decltype(A)>, 128, A, B, C);
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          erg[i] = C[i];
        }
      }));
    } else {
      hpx::cout << "ERROR: Slice 1 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 1 was not created properly");
    }

    auto slice_fut2 = agg_exec.request_executor_slice();
    if (slice_fut2.has_value()) {
      slices_done_futs.emplace_back(slice_fut2.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(slice_exec.number_slices * 128,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = slice_exec.id + 1;
          B[i] = 2 * slice_exec.id;
        }

        // Run add function
        auto kernel_fut = slice_exec.async(add<decltype(A)>, 128, A, B, C);
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          erg[i] = C[i];
        }
      }));
    } else {
      hpx::cout << "ERROR: Slice 2 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 2 was not created properly");
    }

    auto slice_fut3 = agg_exec.request_executor_slice();
    if (slice_fut3.has_value()) {
      slices_done_futs.emplace_back(slice_fut3.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(slice_exec.number_slices * 128,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = slice_exec.id + 1;
          B[i] = 2 * slice_exec.id;
        }

        // Run add function
        auto kernel_fut = slice_exec.async(add<decltype(A)>, 128, A, B, C);
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          erg[i] = C[i];
        }
      }));
    } else {
      hpx::cout << "ERROR: Slice 3 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 3 was not created properly");
    }

    auto slice_fut4 = agg_exec.request_executor_slice();
    if (slice_fut4.has_value()) {
      slices_done_futs.emplace_back(slice_fut4.value().then([&erg](auto &&fut) {
        auto slice_exec = fut.get();
        // Get slice allocator
        auto alloc =
            slice_exec.template make_allocator<float, std::allocator<float>>();
        // Get slice buffers
        std::vector<float, decltype(alloc)> A(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> B(slice_exec.number_slices * 128,
                                              float{}, alloc);
        std::vector<float, decltype(alloc)> C(slice_exec.number_slices * 128,
                                              float{}, alloc);
        // Fill slice buffers
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          A[i] = slice_exec.id + 1;
          B[i] = 2 * slice_exec.id;
        }

        // Run add function
        auto kernel_fut = slice_exec.async(add<decltype(A)>, 128, A, B, C);
        // Sync immediately
        kernel_fut.get();

        // Write results into erg buffer
        for (int i = slice_exec.id * 128; i < (slice_exec.id + 1) * 128; i++) {
          erg[i] = C[i];
        }
      }));
    } else {
      hpx::cout << "ERROR: Slice 4 was not created properly" << std::endl;
      throw std::runtime_error("ERROR: Slice 4 was not created properly");
    }
    hpx::cout << "Requested all executors!" << std::endl;
    hpx::cout << "Realizing by requesting final fut..." << std::endl;
    auto final_fut = hpx::lcos::when_all(slices_done_futs);
    final_fut.get();
    hpx::cout << "Number add_launches=" << add_launches
              << std::endl;
    assert(add_launches == 1);

    hpx::cout << "Checking erg: " << std::endl;
    for (int slice = 0; slice < 4; slice++) {
      for (int i = slice * 128; i < (slice + 1) * 128; i++) {
        assert(erg[i] == 3 * slice + 1);
        hpx::cout << erg[i] << " ";
      }
    }
    hpx::cout << std::endl;
  }
  hpx::cout << std::endl;

  hpx::cout << "Done!" << std::endl;
  hpx::cout << std::endl;
}

//===============================================================================
//===============================================================================
int hpx_main(int argc, char *argv[]) {
  // Init parameters
  std::string scenario{};
  std::string filename{};
  {
    try {
      boost::program_options::options_description desc{"Options"};
      desc.add_options()(
          "help",
          "Help screen")("scenario",
                         boost::program_options::value<std::string>(&scenario)
                             ->default_value("all"),
                         "Which scenario to run [sequential_test, "
                         "interruption_test, failure_test, pointer_add_test, "
                         "references_add_test, all]")("outputfile",
                                                      boost::program_options::
                                                          value<std::string>(
                                                              &filename)
                                                              ->default_value(
                                                                  ""),
                                                      "Redirect stdout/stderr "
                                                      "to this file");

      boost::program_options::variables_map vm;
      boost::program_options::parsed_options options =
          parse_command_line(argc, argv, desc);
      boost::program_options::store(options, vm);
      boost::program_options::notify(vm);

      if (vm.count("help") == 0u) {
        hpx::cout << "Running with parameters:" << std::endl
                  << "--scenario=" << scenario << std::endl
                  << "--outputfile=" << filename << std::endl;
      } else {
        hpx::cout << desc << std::endl;
        return hpx::finalize();
      }
    } catch (const boost::program_options::error &ex) {
      hpx::cout << "CLI argument problem found: " << ex.what() << '\n';
    }
    if (!filename.empty()) {
      freopen(filename.c_str(), "w", stdout); // NOLINT
      freopen(filename.c_str(), "w", stderr); // NOLINT
    }
  }
  if (scenario != "sequential_test" && scenario != "interruption_test" &&
      scenario != "failure_test" && scenario != "pointer_add_test" &&
      scenario != "references_add_test" && scenario != "all") {
    hpx::cout << "ERROR: Invalid scenario specified (see --help)" << std::endl;
    return hpx::finalize();
  }

  stream_pool::init<hpx::cuda::experimental::cuda_executor,
                    round_robin_pool<hpx::cuda::experimental::cuda_executor>>(
      8, 0, false);
  stream_pool::init<Dummy_Executor, round_robin_pool<Dummy_Executor>>(8);

  stream_pool::init<Aggregated_Executor<Dummy_Executor>,
                    round_robin_pool<Aggregated_Executor<Dummy_Executor>>>(
      8, 4, Aggregated_Executor_Modes::STRICT);
  /*hpx::cuda::experimental::cuda_executor executor1 =
      std::get<0>(stream_pool::get_interface<
                  hpx::cuda::experimental::cuda_executor,
                  round_robin_pool<hpx::cuda::experimental::cuda_executor>>());*/

  // Basic tests:
  if (scenario == "sequential_test" || scenario == "all") {
    sequential_test();
  }
  if (scenario == "interruption_test" || scenario == "all") {
    interruption_test();
  }
  if (scenario == "pointer_add_test" || scenario == "all") {
    pointer_add_test();
  }
  if (scenario == "references_add_test" || scenario == "all") {
    references_add_test();
  }
  // Test that checks failure detection in case of wrong usage (missmatching
  // calls/types/values)
  if (scenario == "failure_test" || scenario == "all") {
    failure_test();
  }
  // Flush outout and wait a second for the (non hpx::cout) output to have it in the correct
  // order for the ctests
  std::flush(hpx::cout);
  sleep(1);

  recycler::force_cleanup(); // Cleanup all buffers and the managers
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  std::vector<std::string> cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, cfg);
}
