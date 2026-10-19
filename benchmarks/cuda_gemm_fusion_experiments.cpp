#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

#include "gemm_cpu_utils.hpp"
#include "cuda_gemm_kernels.hpp"

// ------------------------------------------------------------
// Computes:
//   C = alpha * A * B + beta * C
// Matrix layouts:
//   A: M x K, row-major
//   B: K x N, row-major
//   C: M x N, row-major
// ------------------------------------------------------------

// serial / parallel extra argument: Number tasks
// Top-down, updown extra argument: Depth and structure (number childs per node -- lets fix this to 2 for now)
enum graph_mode {scalar = 0, serial, parallel, top_down, bottom_up};
enum fusion_mode {none = 0, cuda_graph, cppuddle};

// hpx::future<void> kernel_fusion_benchmark_iteration(const size_t utilized_implementation_id,
//     const double alpha, const double beta, const std::vector<double> &hA, const std::vector<double> &hB,
//     std::vector<double> &hC, double *dA, double *dB, double *dC, const std::vector<double> &hC_comparison,
//     const size_t M, const size_t N, const size_t K, cublasHandle_t &handle, const graph_mode utilized_graph_mode,
//                              const size_t extra_graph_parameter, const bool postprocessing_active);
hpx::future<void> kernel_fusion_benchmark_iteration(const size_t utilized_implementation_id,
    const double alpha, const double beta, const std::vector<double> &hA, const std::vector<double> &hB,
    std::vector<double> &hC, double *dA, double *dB, double *dC, const std::vector<double> &hC_comparison,
    const size_t M, const size_t N, const size_t K, cublasHandle_t &handle, const graph_mode utilized_graph_mode,
    const size_t extra_graph_parameter, const bool postprocessing_active);

void kernel_fusion_benchmark(const size_t utilized_implementation_id, const std::vector<double> &hA, const std::vector<double> &hB,
                             std::vector<double> &hC, const std::vector<double> &hC_comparison,
                             const size_t M, const size_t N, const size_t K, const size_t num_repetitions, const graph_mode utilized_graph_mode,
                             const size_t extra_graph_parameter, const bool postprocessing_active) {
  const fusion_mode utilized_fusion_approach = none;

  const double flops = 2.0 * static_cast<double>(M)
                     * static_cast<double>(N)
                     * static_cast<double>(K);
  const double alpha = 1.0;
  const double beta  = 0.0;
  const size_t sizeA = M * K;
  const size_t sizeB = K * N;
  const size_t sizeC = M * N;
  const size_t bytesA = sizeA * sizeof(double);
  const size_t bytesB = sizeB * sizeof(double);
  const size_t bytesC = sizeC * sizeof(double);
  double* dA = nullptr;
  double* dB = nullptr;
  double* dC = nullptr;
  cudaMalloc(&dA, bytesA);
  cudaMalloc(&dB, bytesB);
  cudaMalloc(&dC, bytesC);
  cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice);
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N); // first pass required -- I think there is some setup happening

  std::chrono::time_point start_timer = std::chrono::high_resolution_clock::now();
  // setup cuda graph
  // setup executor pool
  // setup multiple cuda streams
  // setup buffers and copu initial data
  // repeat experiment n times (otherwise the cuda graphs will lose badly)
  // ... needs separate methods (cannot extend the recursion over the loop
  auto loop_fut = hpx::make_ready_future<void>();
  for (size_t it = 0; it < num_repetitions; it++) {
    // kernel_fusion_benchmark_iteration(utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
    //     dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter, postprocessing_active);
    // auto loop_fut = hpx::async(
    //     hpx::launch::async_policy(hpx::threads::thread_priority::default_, // maybe we need a generic policy for non-cublas
    //                               hpx::threads::thread_stacksize::large),
    //     &kernel_fusion_benchmark_iteration, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
    //     dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter, postprocessing_active);
    // loop_fut.get();
    // auto loop_fut = hpx::async(
    //     &kernel_fusion_benchmark_iteration, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
    //     dA, dB, dC, hC_comparison, M, N, K, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
    // loop_fut.get();

    loop_fut = loop_fut.then(
        [&](auto && fut) {
        // fut.get();
        return kernel_fusion_benchmark_iteration(utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
        dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter, postprocessing_active);
        });
    //
    // loop_fut = hpx::dataflow([&](auto void_val) {
    //     kernel_fusion_benchmark_iteration(utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
    //     dA, dB, dC, hC_comparison, M, N, K, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
    //     }, loop_fut);
    // loop_fut.get();
    // hpx::post(
    //     &kernel_fusion_benchmark_iteration, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
    //     dA, dB, dC, hC_comparison, M, N, K, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
    // stack size is a problem with the hpx task
    // can be fixed with:
    // --hpx:ini=hpx.stacks.small_size=0x100000.
    
    // kernel_fusion_benchmark_iteration(utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
    //     dA, dB, dC, hC_comparison, M, N, K, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  }
  loop_fut.get();
  cudaGetLastError();
  cudaDeviceSynchronize();
  // cudaGetLastError();
  // cudaDeviceSynchronize();

  auto end_timer = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_timer - start_timer;
  double total_seconds = elapsed.count();
  double avg_seconds = total_seconds / static_cast<double>(num_repetitions);
  if (utilized_graph_mode == graph_mode::serial)
    avg_seconds = avg_seconds / static_cast<double>(extra_graph_parameter);
  double gflops = flops / avg_seconds / 1.0e9;
  std::cout << "-> Average Runtime: " << avg_seconds * 1.0e3 << "ms\n";
  std::cout << "-> Performance: " << gflops << " GFLOP/s";
  std::cout << std::endl;

}
// hpx::future<void> kernel_fusion_benchmark_iteration(const size_t utilized_implementation_id,
//     const double alpha, const double beta, const std::vector<double> &hA, const std::vector<double> &hB,
//     std::vector<double> &hC, double *dA, double *dB, double *dC, const std::vector<double> &hC_comparison,
//     const size_t M, const size_t N, const size_t K, cublasHandle_t &handle, const graph_mode utilized_graph_mode,
//     const size_t extra_graph_parameter, const bool postprocessing_active) {
hpx::future<void> kernel_fusion_benchmark_iteration(const size_t utilized_implementation_id,
    const double alpha, const double beta, const std::vector<double> &hA, const std::vector<double> &hB,
    std::vector<double> &hC, double *dA, double *dB, double *dC, const std::vector<double> &hC_comparison,
    const size_t M, const size_t N, const size_t K, cublasHandle_t &handle, const graph_mode utilized_graph_mode,
    const size_t extra_graph_parameter, const bool postprocessing_active) {

  // auto policy = hpx::launch::async_policy(hpx::threads::thread_priority::default_, 
  //                           hpx::threads::thread_stacksize::large);

  hpx::future<void> ret_fut = hpx::make_ready_future<void>();

  // if graph == parallel
  // Launch one other kernel recursively for <number_kernels>
  // return when-all future
  // if (utilized_graph_mode == graph_mode::parallel) {
  //   if (extra_graph_parameter > 1) {
  //     ret_fut = hpx::async(policy, &kernel_fusion_benchmark_iteration, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
  //         dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
  //   }
  // }

  // if graph == bottom-up
  // Launch two more kernels -- wait until they're done (get), and only then launch this one)
  // do not return future
  // if (utilized_graph_mode == graph_mode::bottom_up) {
  //   if (extra_graph_parameter > 1) {
  //     auto fut1 = hpx::async(policy, &kernel_fusion_benchmark_iteration, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
  //         dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
  //     auto fut2 = hpx::async(policy, &kernel_fusion_benchmark_iteration, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
  //         dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
  //     ret_fut = hpx::when_all(fut1, fut2);
  //   }
  // }

  if (utilized_implementation_id == 0) {
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    // start_timer = std::chrono::high_resolution_clock::now();
    // preprocessing?
    dgemm_naive_unoptimized_v0<<<grid, block>>>(
      M, N, K, alpha, dA, dB, beta, dC);
    // postprocessing
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant 0 faulty!" << std::endl;
  } else if (utilized_implementation_id == 1) {
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    dgemm_naive_v1<<<grid, block>>>(
      M, N, K, alpha, dA, dB, beta, dC);
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant 1 faulty!" << std::endl;
  } else if (utilized_implementation_id == 2) {
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid(
      (N + TILE - 1) / TILE,
      (M + TILE - 1) / TILE
    );
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    dgemm_shared_tiled_v2<TILE><<<grid, block>>>(
      M, N, K, alpha, dA, dB, beta, dC);
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant 2 faulty!" << std::endl;
  } else if (utilized_implementation_id == 3) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    dgemm_register_tiled_v3<BM, BN, BK, TM, TN><<<grid, block>>>(
      M, N, K, alpha, dA, dB, beta, dC);
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant 3 faulty!" << std::endl;
  } else if (utilized_implementation_id == 4) {
    constexpr int BM = 64, BN = 32, BK = 16, TM = 4, TN = 2;
    dim3 block(BN / TN, BM / TM);
    dim3 grid(
      (N + BN - 1) / BN,
      (M + BM - 1) / BM
    );
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    dgemm_v4_shared_register_tiled<BM, BN, BK, TM, TN>
      <<<grid, block>>>(
        M, N, K, alpha, dA, dB, beta, dC);
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant 4 faulty!" << std::endl;
  } else if (utilized_implementation_id == 5) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    dgemm_v5_vectorized_register_tiled<BM, BN, BK, TM, TN>
      <<<grid, block>>>(
      M, N, K, alpha, dA, dB, beta, dC);
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant 5 faulty!" << std::endl;
  } else if (utilized_implementation_id == 6) {
    constexpr int BM = 64, BN = 64, BK = 16;
    if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
      std::cerr << "dgemm_v5b_fp64_wmma_tiled requires "
                << "M %% 64 == 0, N %% 64 == 0, K %% 16 == 0\n"
                << std::endl;
      exit(0);
    }
    dim3 block(512);
    dim3 grid(N / BN, M / BM);
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    dgemm_v6_fp64_wmma_tiled<<<grid,  block>>>(
      M, N, K, alpha, dA, dB, beta, dC);
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant 6 faulty!" << std::endl;
  } else if (utilized_implementation_id == 7) {
    constexpr int BM = 64, BN = 64, BK = 16;
    if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
      std::cerr << "dgemm_v7_fp64_wmma_64x64_8warps requires "
                << "M %% 64 == 0, N %% 64 == 0, K %% 16 == 0\n"
                << std::endl;
      exit(0);
    }
    dim3 block(256);
    dim3 grid(N / BN, M / BM);
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    dgemm_v7_fp64_wmma_64x64_8warps<BM, BN, BK>
      <<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant 7 faulty!" << std::endl;
  } else if (utilized_implementation_id == 8) {  // CUBLAS version
    if (postprocessing_active)
      reset_C_gpu(M, N, dC);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    if (postprocessing_active && !check_gpu_results(M, N, dC, hC_comparison.data())) 
      std::cout << "-> ERROR: Variant CUBLAS faulty!" << std::endl;
  } else if (utilized_implementation_id == 9) { // CPU version
    reset_C_cpu(M, N, hC.data());
    dgemm_naive_cpu(
    M, N, K, alpha, hA.data(), hB.data(), beta, hC.data());
    if (postprocessing_active && !compare_matrices(M, N, hC.data(), hC_comparison.data())) 
      std::cout << "-> ERROR: Variant CPU faulty!" << std::endl;
  }

  if (utilized_graph_mode == graph_mode::serial) {
    if (extra_graph_parameter > 1) {
      ret_fut = ret_fut.then(
          [=, &handle, &hA, &hB, &hC, &hC_comparison](auto && fut) {
        kernel_fusion_benchmark_iteration(utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
        dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
          });
      // hpx::async(&kernel_fusion_benchmark_iteration, std::ref(previous_it),  utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
      //     dA, dB, dC, hC_comparison, M, N, K, utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
      // std::cout << extra_graph_parameter << " " << postprocessing_active << std::endl;
      // previous_it.then([=, &hA, &hB, &hC, &hC_comparison](auto && fut) {
      //   kernel_fusion_benchmark_iteration(fut, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
      //        dA, dB, dC, hC_comparison, M, N, K, utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
      //    });
      // kernel_fusion_benchmark_iteration(previous_it, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
      //     dA, dB, dC, hC_comparison, M, N, K, utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
      // ret_fut = hpx::make_ready_future<void>();
    }
  }
  
  // if (utilized_graph_mode == graph_mode::top_down) {
  //   if (extra_graph_parameter > 1) {
  //     auto fut1 = hpx::async(policy, &kernel_fusion_benchmark_iteration, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
  //         dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
  //     auto fut2 = hpx::async(policy, &kernel_fusion_benchmark_iteration, utilized_implementation_id, alpha, beta, hA, hB, std::ref(hC),
  //         dA, dB, dC, hC_comparison, M, N, K, std::ref(handle), utilized_graph_mode, extra_graph_parameter - 1, postprocessing_active);
  //     ret_fut = hpx::when_all(fut1, fut2);
  //   }
  // }
  return ret_fut;
}

int hpx_main(int argc, char *argv[]) {
  std::cout << "Running benchmark with " << hpx::get_num_worker_threads() << " HPX worker threads!" << std::endl;
  size_t M = 1024;
  size_t N = 1024;
  size_t K = 1024;

  if (argc >= 4) {
    M = std::stoull(argv[1]);
    N = std::stoull(argv[2]);
    K = std::stoull(argv[3]);
  }

  std::cout << "Using Matrix Dimensions: ";
  std::cout << "M = " << M << ", N = " << N << ", K = " << K << "\n";

  const double alpha = 1.0;
  const double beta  = 0.0;

  size_t sizeA = static_cast<size_t>(M) * K;
  size_t sizeB = static_cast<size_t>(K) * N;
  size_t sizeC = static_cast<size_t>(M) * N;

  size_t bytesA = sizeA * sizeof(double);
  size_t bytesB = sizeB * sizeof(double);
  size_t bytesC = sizeC * sizeof(double);

  std::vector<double> hA(sizeA);
  std::vector<double> hB(sizeB);
  std::vector<double> hC(sizeC);
  std::vector<double> hC_comparison(sizeC);

  const unsigned long long seed = 42;
  std::mt19937_64 rng(seed);
  fill_random(hA, rng);
  fill_random(hB, rng);
  fill_random(hC, rng);
  reset_C_cpu(M, N, hC_comparison.data());

  std::string alpha_string =  std::to_string(alpha);
  std::replace(alpha_string.begin(), alpha_string.end(), '.', '-');
  std::string beta_string =  std::to_string(beta);
  std::replace(beta_string.begin(), beta_string.end(), '.', '-');
  std::string reference_data_filename = std::string("reference_") + std::to_string(seed) + 
    std::string("_") + std::to_string(M) + std::string("_") + std::to_string(N) + std::string("_") + std::to_string(K) +
    std::string("_") + alpha_string + std::string("_") + beta_string +
    std::string(".txt");
  std::cout << "Checking if reference file for this the given M, N, K, alpha, beta and seed " << seed << " exists..." << std::endl;
  std::cout << "Filename format: reference_<seed>_<M>_<N>_<K>_<alpha>_<beta>.txt (alpha and beta use - instead of the decimal point)" << std::endl;
  bool generate_results = true;
  if (std::filesystem::exists(reference_data_filename)) {
    std::cout << "Reference file " << reference_data_filename << " found!" << std::endl;
    std::cout << "Loading reference file ... " << std::endl;
    try {
      std::ifstream in(reference_data_filename);
      // C: M x N, row-major
      for (auto row = 0; row < M; row++) {
        for (auto column = 0; column < N; column++) {
          in >> hC_comparison[row * N + column];
        }
      }
      in.close();
      generate_results = false;
      std::cout << "Reference data loaded! Continuing with benchmarks..." << std::endl;
    } catch (...) {
      std::cerr << "Unknown error when reading reference data" << reference_data_filename << " !" << std::endl;
      std::cout << "Continuing with generating input data after failed reading!" << std::endl;
    }

  } else {
    std::cout << "Reference file " << reference_data_filename << " not found in current directory!" << std::endl;
  }

  if (generate_results) {
    std::cout << "Generating reference data on CPU... " << std::endl;
    dgemm_naive_cpu(
      M, N, K, alpha, hA.data(), hB.data(), beta, hC_comparison.data());

    std::cout << "Reference data generated! Storing in " << reference_data_filename << "..." << std::endl;
    try {
      std::ofstream out(reference_data_filename);
      out << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);
      // C: M x N, row-major
      for (auto row = 0; row < M; row++) {
        for (auto column = 0; column < N; column++) {
          out << hC_comparison[row * N + column] << " ";
        }
        out << std::endl;
      }
      out.close();
      std::cout << "Reference data stored! Continuing with benchmarks..." << std::endl;
    } catch (...) {
      std::cerr << "Unknown error when writing reference data" << reference_data_filename << " !" << std::endl;
    }

  }

  double* dA = nullptr;
  double* dB = nullptr;
  double* dC = nullptr;

  cudaMalloc(&dA, bytesA);
  cudaMalloc(&dB, bytesB);
  cudaMalloc(&dC, bytesC);

  cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice);

  dim3 block_v1(16, 16);
  dim3 grid_v1(
      (N + block_v1.x - 1) / block_v1.x,
      (M + block_v1.y - 1) / block_v1.y
  );

  // ------------------------------------------------------------
  // FLOP estimate
  // For GEMM:
  //   C = A * B
  //   FLOPs = 2 * M * N * K
  // ------------------------------------------------------------
  double flops = 2.0 * static_cast<double>(M)
                     * static_cast<double>(N)
                     * static_cast<double>(K);
  std::cout << "Estimated FLOPs:     " << flops << "\n";

  // ------------------------------------------------------------
  // Timed execution
  // Results
  // ------------------------------------------------------------
  std::cout << "\nResults\n";
  std::cout << "-------\n\n";
  const size_t  num_repetitions = 10;
  const graph_mode utilized_graph_mode = serial;  
  const size_t extra_graph_parameter = 10;
  const bool postprocessing_active = false;


  // ============================================================
  // CPU Comparison variant (datA)
  // ============================================================
  std::cout << "Variant Benchmark CPU: \n";
  kernel_fusion_benchmark(9, hA, hB, hC, hC_comparison, N, M, K, 0, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark CPU end \n" << std::endl;
  // std::cout << "Variant CPU: Naive \n";

  // auto start_cpu = std::chrono::high_resolution_clock::now();
  // dgemm_naive_cpu(
  //   M, N, K, alpha, hA.data(), hB.data(), beta, hC_comparison.data());
  // auto end_cpu = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
  // double total_seconds_cpu = elapsed_cpu.count();
  // double avg_seconds_cpu = total_seconds_cpu / static_cast<double>(1);
  // double gflops_cpu = flops / avg_seconds_cpu / 1.0e9;
  // std::cout << "-> Runtime: " << avg_seconds_cpu * 1.0e3 << "ms\n";
  // std::cout << "-> Variant CPU performance: " << gflops_cpu << " GFLOP/s\n";
  // std::cout << std::endl;

  // ============================================================
  // Variant 0 launch configuration
  // ============================================================
  // std::cout << "Variant Benchmark 0: Naive \n";
  // kernel_fusion_benchmark(0, hA, hB, hC, hC_comparison, N, M, K);
  // std::cout << "Variant Benchmark 0 end \n";
  std::cout << "Variant 0 Naive: \n";
  kernel_fusion_benchmark(0, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant 0 Naive \n" << std::endl;

  // std::cout << "Variant 0: Naive \n";
  // auto start_v0 = std::chrono::high_resolution_clock::now();
  // for (int repeat = 0; repeat < num_repeats; ++repeat) {
  //   dgemm_naive_unoptimized_v0<<<grid_v1, block_v1>>>(
  //     M, N, K, alpha, dA, dB, beta, dC);
  // }
  // cudaGetLastError();
  // cudaDeviceSynchronize();
  // auto end_v0 = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_v0 = end_v0 - start_v0;
  // double total_seconds_v0 = elapsed_v0.count();
  // double avg_seconds_v0 = total_seconds_v0 / static_cast<double>(num_repeats);
  // double gflops_v0 = flops / avg_seconds_v0 / 1.0e9;
  // if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //   std::cout << "-> Results correct!" << std::endl;
  // else
  //   std::cout << "-> ERROR: Variant 0 faulty!" << std::endl;
  // std::cout << "-> Average Runtime: " << avg_seconds_v0 * 1.0e3 << "ms\n";
  // std::cout << "-> Performance: " << gflops_v0 << " GFLOP/s\n";
  // std::cout << std::endl;
    

  // ============================================================
  // Variant 1 launch configuration
  // ============================================================
  std::cout << "Variant Benchmark 1: \n";
  std::cout << "-> Description: Naive but with optimized access pattern\n";
  kernel_fusion_benchmark(1, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark 1 \n" << std::endl;

  // std::cout << "Variant 1: Naive but with optimized access pattern\n";
  // auto start_v1 = std::chrono::high_resolution_clock::now();
  // for (int repeat = 0; repeat < num_repeats; ++repeat) {
  //   dgemm_naive_v1<<<grid_v1, block_v1>>>(
  //     M, N, K, alpha, dA, dB, beta, dC);
  // }
  // cudaGetLastError();
  // cudaDeviceSynchronize();
  // auto end_v1 = std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> elapsed_v1 = end_v1 - start_v1;
  // double total_seconds_v1 = elapsed_v1.count();
  // double avg_seconds_v1 = total_seconds_v1 / static_cast<double>(num_repeats);
  // double gflops_v1 = flops / avg_seconds_v1 / 1.0e9;
  // std::cout << "-> Block: " << block_v1.x << " x " << block_v1.y << "\n";
  // std::cout << "-> Grid:  " << grid_v1.x  << " x " << grid_v1.y  << "\n";
  // if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //   std::cout << "-> Results correct!" << std::endl;
  // else
  //   std::cout << "-> ERROR: Variant 1 faulty!" << std::endl;
  // std::cout << "-> average runtime: " << avg_seconds_v1 * 1.0e3 << "ms\n";
  // std::cout << "-> performance: " << gflops_v1 << " GLOP/s\n";
  // std::cout << std::endl;

  // // Optionally copy result back to host
  // cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost);


  // ============================================================
  // Variant 2 launch configuration
  // ============================================================
  std::cout << "Variant Benchmark 2: \n";
  std::cout << "-> Description: Shared-memory tiled\n";
  kernel_fusion_benchmark(2, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark 2 \n" << std::endl;

  // std::cout << "Variant 2: shared-memory tiled\n";

  // constexpr int TILE_V2 = 32;
  // dim3 block_v2(TILE_V2, TILE_V2);
  // dim3 grid_v2(
  //   (N + TILE_V2 - 1) / TILE_V2,
  //   (M + TILE_V2 - 1) / TILE_V2
  // );
  // auto start_v2 = std::chrono::high_resolution_clock::now();
  // for (int repeat = 0; repeat < num_repeats; ++repeat) {
  //   dgemm_shared_tiled_v2<TILE_V2><<<grid_v2, block_v2>>>(
  //     M, N, K, alpha, dA, dB, beta, dC);
  // }
  // cudaGetLastError();
  // cudaDeviceSynchronize();
  // auto end_v2 = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_v2 = end_v2 - start_v2;
  // double total_seconds_v2 = elapsed_v2.count();
  // double avg_seconds_v2 = total_seconds_v2 / static_cast<double>(num_repeats);
  // double gflops_v2 = flops / avg_seconds_v2 / 1.0e9;
  // std::cout << "-> Block: " << block_v2.x << " x " << block_v2.y << "\n";
  // std::cout << "-> Grid:  " << grid_v2.x  << " x " << grid_v2.y  << "\n";
  // if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //   std::cout << "-> Results correct!" << std::endl;
  // else
  //   std::cout << "-> ERROR: Variant 2 faulty!" << std::endl;
  // std::cout << "-> average runtime: " << avg_seconds_v2 * 1.0e3 << "ms\n";
  // std::cout << "-> performance: " << gflops_v2 << " GLOP/s\n";
  // std::cout << std::endl;


  // ============================================================
  // Variant 3 launch configuration
  // ============================================================
  std::cout << "Variant Benchmark 3: \n";
  std::cout << "-> Description: Register tiled\n";
  kernel_fusion_benchmark(3, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark end 3 \n" << std::endl;

  // std::cout << "Variant 3: register tiled\n";
  // constexpr int BM_v3 = 64;
  // constexpr int BN_v3 = 64;
  // constexpr int BK_v3 = 8;
  // constexpr int TM_v3 = 4;
  // constexpr int TN_v3 = 4;

  // static_assert(BM_v3 % TM_v3 == 0, "BM must be divisible by TM");
  // static_assert(BN_v3 % TN_v3 == 0, "BN must be divisible by TN");

  // dim3 block_v3(BN_v3 / TN_v3, BM_v3 / TM_v3);
  // dim3 grid_v3(
  //   (N + BN_v3 - 1) / BN_v3,
  //   (M + BM_v3 - 1) / BM_v3
  // );
  // auto start_v3 = std::chrono::high_resolution_clock::now();
  // for (int repeat = 0; repeat < num_repeats; ++repeat) {
  //   dgemm_register_tiled_v3<BM_v3, BN_v3, BK_v3, TM_v3, TN_v3><<<grid_v3, block_v3>>>(
  //     M, N, K, alpha, dA, dB, beta, dC);
  // }
  // cudaGetLastError();
  // cudaDeviceSynchronize();
  // auto end_v3 = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_v3 = end_v3 - start_v3;
  // double total_seconds_v3 = elapsed_v3.count();
  // double avg_seconds_v3 = total_seconds_v3 / static_cast<double>(num_repeats);
  // double gflops_v3 = flops / avg_seconds_v3 / 1.0e9;

  // // std::cout << "-> BM = " << BM_v3 << ", BN = " << BN_v3
  // //           << ", BK = " << BK_v3 << "\n";
  // // std::cout << "-> TM = " << TM_v3 << ", TN = " << TN_v3 << "\n";
  // std::cout << "-> Block: " << block_v3.x << " x " << block_v3.y << "\n";
  // std::cout << "-> Grid:  " << grid_v3.x  << " x " << grid_v3.y  << "\n";
  // if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //   std::cout << "-> Results correct!" << std::endl;
  // else
  //   std::cout << "-> ERROR: Variant 3 faulty!" << std::endl;
  // std::cout << "-> average runtime: " << avg_seconds_v3 * 1.0e3 << "ms\n";
  // std::cout << "-> performance: " << gflops_v3 << " GLOP/s\n";
  // std::cout << std::endl;

  // ============================================================
  // Variant 4 launch configuration
  // ============================================================
  std::cout << "Variant Benchmark 4: \n";
  std::cout << "-> Description: Shared register tiled with better ccnfig\n";
  kernel_fusion_benchmark(4, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark 4 end \n" << std::endl;
  // std::cout << "Variant 4: shared register tiled with better ccnfig\n";

  // constexpr int BM_v4 = 64;
  // constexpr int BN_v4 = 32;
  // constexpr int BK_v4 = 16;
  // constexpr int TM_v4 = 4;
  // constexpr int TN_v4 = 2;

  // dim3 block_v4(BN_v4 / TN_v4, BM_v4 / TM_v4);
  // dim3 grid_v4(
  //   (N + BN_v4 - 1) / BN_v4,
  //   (M + BM_v4 - 1) / BM_v4
  // );
  // auto start_v4 = std::chrono::high_resolution_clock::now();
  // for (int repeat = 0; repeat < num_repeats; ++repeat) {
  //   dgemm_v4_shared_register_tiled<BM_v4, BN_v4, BK_v4, TM_v4, TN_v4>
  //     <<<grid_v4, block_v4>>>(
  //       M, N, K, alpha, dA, dB, beta, dC);
  // }
  // cudaGetLastError();
  // cudaDeviceSynchronize();
  // auto end_v4 = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_v4 = end_v4 - start_v4;
  // double total_seconds_v4 = elapsed_v4.count();
  // double avg_seconds_v4 = total_seconds_v4 / static_cast<double>(num_repeats);
  // double gflops_v4 = flops / avg_seconds_v4 / 1.0e9;

  // // std::cout << "-> BM = " << BM_v4 << ", BN = " << BN_v4
  // //           << ", BK = " << BK_v4 << "\n";
  // // std::cout << "-> TM = " << TM_v4 << ", TN = " << TN_v4 << "\n";
  // std::cout << "-> Block: " << block_v4.x << " x " << block_v4.y << "\n";
  // std::cout << "-> Grid:  " << grid_v4.x  << " x " << grid_v4.y  << "\n";
  // if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //   std::cout << "-> Results correct!" << std::endl;
  // else
  //   std::cout << "-> ERROR: Variant 4 faulty!" << std::endl;
  // std::cout << "-> average runtime: " << avg_seconds_v4 * 1.0e3 << "ms\n";
  // std::cout << "-> performance: " << gflops_v4 << " GLOP/s\n";
  // std::cout << std::endl;

  // ============================================================
  // Variant 5 launch configuration
  // ============================================================
  std::cout << "Variant Benchmark 5: \n";
  std::cout << "-> Description: Vectorized shared register tiled\n";
  kernel_fusion_benchmark(5, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark 5 end \n" << std::endl;
  // std::cout << "Variant 5: vectorized shared register tiled\n";

  // constexpr int BM_v5 = 64;
  // constexpr int BN_v5 = 64;
  // constexpr int BK_v5 = 8;
  // constexpr int TM_v5 = 4;
  // constexpr int TN_v5 = 4;

  // dim3 block_v5(BN_v5 / TN_v5, BM_v5 / TM_v5);
  // dim3 grid_v5(
  //   (N + BN_v5 - 1) / BN_v5,
  //   (M + BM_v5 - 1) / BM_v5
  // );
  // auto start_v5 = std::chrono::high_resolution_clock::now();
  // for (int repeat = 0; repeat < num_repeats; ++repeat) {
  //   dgemm_v5_vectorized_register_tiled<BM_v5, BN_v5, BK_v5, TM_v5, TN_v5>
  //     <<<grid_v5, block_v5>>>(
  //       M, N, K, alpha, dA, dB, beta, dC);
  // }
  // cudaGetLastError();
  // cudaDeviceSynchronize();
  // auto end_v5 = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_v5 = end_v5 - start_v5;
  // double total_seconds_v5 = elapsed_v5.count();
  // double avg_seconds_v5 = total_seconds_v5 / static_cast<double>(num_repeats);
  // double gflops_v5 = flops / avg_seconds_v5 / 1.0e9;

  // // std::cout << "-> BM = " << BM_v5 << ", BN = " << BN_v5
  // //           << ", BK = " << BK_v5 << "\n";
  // // std::cout << "-> TM = " << TM_v5 << ", TN = " << TN_v5 << "\n";
  // std::cout << "-> Block: " << block_v5.x << " x " << block_v5.y << "\n";
  // std::cout << "-> Grid:  " << grid_v5.x  << " x " << grid_v5.y  << "\n";
  // if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //     std::cout << "-> Results correct!" << std::endl;
  // else
  //     std::cout << "-> ERROR: Variant 5 faulty!" << std::endl;
  // std::cout << "-> average runtime: " << avg_seconds_v5 * 1.0e3 << "ms\n";
  // std::cout << "-> performance: " << gflops_v5 << " GLOP/s\n";
  // std::cout << std::endl;

  // ============================================================
  // Variant 6 launch configuration
  // ============================================================
  std::cout << "Variant Benchmark 6: \n";
  std::cout << "-> Description: WMMA\n";
  kernel_fusion_benchmark(6, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark 6 end \n" << std::endl;
  // std::cout << "Variant 6: WMMA\n";
  
  // constexpr int BM_v6 = 64;
  // constexpr int BN_v6 = 64;
  // constexpr int BK_v6 = 16;

  // if ((M % BM_v6) != 0 || (N % BN_v6) != 0 || (K % BK_v6) != 0) {
  //   fprintf(stderr,
  //           "dgemm_v5b_fp64_wmma_tiled requires "
  //           "M %% 64 == 0, N %% 64 == 0, K %% 16 == 0\n");
  //   return 0;
  // }

  // dim3 block_v6(512);
  // dim3 grid_v6(N / BN_v6, M / BM_v6);

  // auto start_v6 = std::chrono::high_resolution_clock::now();
  // for (int repeat = 0; repeat < num_repeats; ++repeat) {
  //   dgemm_v6_fp64_wmma_tiled<<<grid_v6,  block_v6>>>(
  //     M, N, K, alpha, dA, dB, beta, dC);
  // }
  // cudaGetLastError();
  // cudaDeviceSynchronize();
  // auto end_v6 = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_v6 = end_v6 - start_v6;
  // double total_seconds_v6 = elapsed_v6.count();
  // double avg_seconds_v6 = total_seconds_v6 / static_cast<double>(num_repeats);
  // double gflops_v6 = flops / avg_seconds_v6 / 1.0e9;
  // std::cout << "-> Block: " << block_v6.x << " x " << block_v6.y << "\n";
  // std::cout << "-> Grid:  " << grid_v6.x  << " x " << grid_v6.y  << "\n";
  // if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //     std::cout << "-> Results correct!" << std::endl;
  // else
  //     std::cout << "-> ERROR: Variant 6 faulty!" << std::endl;
  // std::cout << "-> average runtime: " << avg_seconds_v6 * 1.0e3 << "ms\n";
  // std::cout << "-> performance: " << gflops_v6 << " GLOP/s\n";
  // std::cout << std::endl;

  // ============================================================
  // Variant 7 launch configuration
  // ============================================================
  std::cout << "Variant Benchmark 7: \n";
  std::cout << "-> Description: Better WMMA\n";
  kernel_fusion_benchmark(7, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark 7 end \n" << std::endl;
  // std::cout << "Variant 7: Better WMMA\n";
  // constexpr int BM_v7 = 64;
  // constexpr int BN_v7 = 64;
  // constexpr int BK_v7 = 16;

  // if ((M % BM_v7) != 0 || (N % BN_v7) != 0 || (K % BK_v7) != 0) {
  //   fprintf(stderr,
  //           "dgemm_v7_fp64_wmma_64x64_8warps requires "
  //           "M %% 64 == 0, N %% 64 == 0, K %% 16 == 0\n");
  //   return 0;
  // }

  // dim3 block_v7(256);
  // dim3 grid_v7(N / BN_v7, M / BM_v7);

  // auto start_v7 = std::chrono::high_resolution_clock::now();
  // for (int repeat = 0; repeat < num_repeats; ++repeat) {
  // dgemm_v7_fp64_wmma_64x64_8warps<BM_v7, BN_v7, BK_v7>
  //   <<<grid_v7, block_v7>>>(M, N, K, alpha, dA, dB, beta, dC);
  // }
  // cudaGetLastError();
  // cudaDeviceSynchronize();
  // auto end_v7 = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_v7 = end_v7 - start_v7;
  // double total_seconds_v7 = elapsed_v7.count();
  // double avg_seconds_v7 = total_seconds_v7 / static_cast<double>(num_repeats);
  // double gflops_v7 = flops / avg_seconds_v7 / 1.0e9;
  // std::cout << "Block: " << block_v7.x << " x " << block_v7.y << "\n";
  // std::cout << "Grid:  " << grid_v7.x  << " x " << grid_v7.y  << "\n";
  // if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //   std::cout << "-> Results correct!" << std::endl;
  // else
  //   std::cout << "-> ERROR: Variant 7 faulty!" << std::endl;
  // std::cout << "-> average runtime: " << avg_seconds_v7 * 1.0e3 << "ms\n";
  // std::cout << "-> performance: " << gflops_v7 << " GLOP/s\n";
  // std::cout << std::endl;


  std::cout << "Variant Benchmark cublas: \n";
  std::cout << "-> Description: CUBLAS: \n";
  kernel_fusion_benchmark(8, hA, hB, hC, hC_comparison, N, M, K, num_repetitions, utilized_graph_mode, extra_graph_parameter, postprocessing_active);
  std::cout << "Variant Benchmark cublas end \n" << std::endl;

  //std::cout << "Variant CUBLAS: \n";
  //cublasHandle_t handle;
  //cublasCreate(&handle);
  //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  //// Optional if you use a specific stream:
  //// cudaStream_t stream;
  //// CHECK_CUDA(cudaStreamCreate(&stream));
  //// CHECK_CUBLAS(cublasSetStream(handle, stream));
  ////cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  //// warmup
  //cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  //cudaGetLastError();
  //cudaDeviceSynchronize();
  //auto start_cublas = std::chrono::high_resolution_clock::now();
  //for (int repeat = 0; repeat < num_repeats; ++repeat) {
  //  // A in cuBLAS sense: B_row interpreted as B^T
  //  // B in cuBLAS sense: A_row interpreted as A^T
  //  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  //}
  //cudaGetLastError();
  //cudaDeviceSynchronize();
  //auto end_cublas = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double> elapsed_cublas = end_cublas - start_cublas;
  //double total_seconds_cublas = elapsed_cublas.count();
  //double avg_seconds_cublas = total_seconds_cublas / static_cast<double>(num_repeats);
  //double gflops_cublas = flops / avg_seconds_cublas / 1.0e9;
  //std::cout << "Variant CUBLAS: \n";
  //if (check_gpu_results(M, N, dC, hC_comparison.data())) 
  //  std::cout << "-> Results correct!" << std::endl;
  //else
  //  std::cout << "-> ERROR: Variant CUBLAS faulty!" << std::endl;
  //std::cout << "-> average runtime: " << avg_seconds_cublas * 1.0e3 << "ms\n";
  //std::cout << "-> performance: " << gflops_cublas << " GLOP/s\n";
  //std::cout << std::endl;

  // ------------------------------------------------------------
  // Cleanup
  // ------------------------------------------------------------
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  hpx::init_params p;
  p.cfg = {"hpx.commandline.allow_unknown=1"};
  return hpx::init(argc, argv, p);
}
