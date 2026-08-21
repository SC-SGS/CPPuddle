#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>


#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

// ------------------------------------------------------------
// Variant 1: naive row-major DGEMM
//
// Computes:
//
//   C = alpha * A * B + beta * C
//
// Matrix layouts:
//
//   A: M x K, row-major
//   B: K x N, row-major
//   C: M x N, row-major
// ------------------------------------------------------------

void dgemm_naive_cpu(
    int M, int N, int K, double alpha, const double* A, const double* B, double beta, double* C) {
  for (auto row = 0; row < M; row++) {
    for (auto column = 0; column < N; column++) {
      double acc = 0.0;
      for (auto entry = 0; entry < K; entry++) {
        acc += A[row * K + entry] * B[column + entry * N];
      }
      C[row * N + column] = alpha * acc + beta * C[row * N + column];
    }
  }
}

bool compare_matrices(int M, int N, double *C_1, double *C_2, double epsilon = 1e-12) {
  bool correct = true;
  for (auto entry = 0; entry < M * N; entry ++) {
    if (std::abs(C_1[entry] - C_2[entry]) > epsilon) {
      correct = false;
      std::cerr << "Matrices C_1 and C_2 differ at row " << entry / N << " col "
                << entry % N << " with C_1 " << C_1[entry] << " and C_2 " << C_2[entry] 
                << std::endl;
    }
  }
  return correct;
}

bool check_gpu_results(int M, int N, double *C_device_results, double *C_reference) {
  std::vector<double> c_host(M*N);
  cudaMemcpy(c_host.data(), C_device_results, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemset(C_device_results, 0, M * N * sizeof(double));
  return compare_matrices(M, N, c_host.data(), C_reference);
}

void reset_C_cpu(int M, int N, double *C) {
  std::memset(C, 0, M * N * sizeof(double));
}


__global__ void dgemm_naive_unoptimized_v0(
    int M, int N, int K, double alpha, const double* A, const double* B, double beta, double* C) {
    // Deliberately bad mapping:
    // threadIdx.x maps to row instead of column.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            double a = A[row * K + k];
            double b = B[k * N + col];
            acc = acc + a * b;
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}
__global__ void dgemm_naive_v1(
    int M, int N, int K, double alpha, const double* __restrict__ A, const double* __restrict__ B,
    double beta, double* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            double a = A[row * K + k];
            double b = B[k * N + col];
            acc = fma(a, b, acc);
        }
        int idx = row * N + col;
        C[idx] = alpha * acc + beta * C[idx];
    }
}

template <int TILE>
__global__ void dgemm_shared_tiled_v2(
    int M, int N, int K, double alpha, const double* __restrict__ A, const double* __restrict__ B,
    double beta, double* __restrict__ C) {

    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    double acc = 0.0;

    for (int kb = 0; kb < K; kb += TILE) {
        int a_col = kb + tx;
        int b_row = kb + ty;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0;
        }
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc = fma(As[ty][kk], Bs[kk][tx], acc);
        }

        __syncthreads();
    }
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = alpha * acc + beta * C[idx];
    }
}


template <int BM, int BN, int BK, int TM, int TN>
__global__ void dgemm_register_tiled_v3(
    int M, int N, int K, double alpha, const double* __restrict__ A, const double* __restrict__ B,
    double beta, double* __restrict__ C) {

    __shared__ double As[BM][BK + 1];
    __shared__ double Bs[BK][BN + 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int thread_row = ty * TM;
    int thread_col = tx * TN;
    double acc[TM][TN];

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    for (int kb = 0; kb < K; kb += BK) {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
            int r = idx / BK;
            int c = idx % BK;
            int global_row = block_row + r;
            int global_col = kb + c;

            if (global_row < M && global_col < K) {
                As[r][c] = A[global_row * K + global_col];
            } else {
                As[r][c] = 0.0;
            }
        }

        for (int idx = tid; idx < BK * BN; idx += num_threads) {
            int r = idx / BN;
            int c = idx % BN;
            int global_row = kb + r;
            int global_col = block_col + c;

            if (global_row < K && global_col < N) {
                Bs[r][c] = B[global_row * N + global_col];
            } else {
                Bs[r][c] = 0.0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a_frag[TM];
            double b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[thread_row + i][kk];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[kk][thread_col + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fma(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = block_row + thread_row + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_col = block_col + thread_col + j;

            if (global_row < M && global_col < N) {
                int idx = global_row * N + global_col;
                C[idx] = alpha * acc[i][j] + beta * C[idx];
            }
        }
    }
}

template <
    int BM = 64,   // block tile size in M dimension
    int BN = 32,   // block tile size in N dimension
    int BK = 16,   // tile size in K dimension
    int TM = 4,    // per-thread tile size in M
    int TN = 2     // per-thread tile size in N
>
__global__ void dgemm_v4_shared_register_tiled(
    int M, int N, int K, double alpha, const double* __restrict__ A, const double* __restrict__ B,
    double beta, double* __restrict__ C) {

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    constexpr int blockDimX = BN / TN;
    constexpr int blockDimY = BM / TM;
    __shared__ double As[BM][BK + 1];
    __shared__ double Bs[BK][BN + 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;
    int localRow = ty * TM;
    int localCol = tx * TN;
    double acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    int tid = ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += numThreads) {
            int r = idx / BK;
            int k = idx % BK;
            int globalRow = blockRow + r;
            int globalK   = k0 + k;

            if (globalRow < M && globalK < K) {
                As[r][k] = A[globalRow * K + globalK];
            } else {
                As[r][k] = 0.0;
            }
        }

        for (int idx = tid; idx < BK * BN; idx += numThreads) {
            int k = idx / BN;
            int c = idx % BN;
            int globalK   = k0 + k;
            int globalCol = blockCol + c;

            if (globalK < K && globalCol < N) {
                Bs[k][c] = B[globalK * N + globalCol];
            } else {
                Bs[k][c] = 0.0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            double aFrag[TM];
            double bFrag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                aFrag[i] = As[localRow + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                bFrag[j] = Bs[k][localCol + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fma(aFrag[i], bFrag[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int globalRow = blockRow + localRow + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int globalCol = blockCol + localCol + j;

            if (globalRow < M && globalCol < N) {
                int idx = globalRow * N + globalCol;
                C[idx] = alpha * acc[i][j] + beta * C[idx];
            }
        }
    }
}

template <
    int BM = 64,
    int BN = 64,
    int BK = 8,
    int TM = 4,
    int TN = 4
>
__global__ void dgemm_v5_vectorized_register_tiled(
    int M, int N, int K, double alpha, const double* __restrict__ A, const double* __restrict__ B,
    double beta, double* __restrict__ C) {

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert(BK % 2 == 0, "BK must be even for double2 loads");
    static_assert(BN % 2 == 0, "BN must be even for double2 loads");
    constexpr int THREADS_X = BN / TN;  // 16
    constexpr int THREADS_Y = BM / TM;  // 16
    __shared__ double As[BM][BK];
    __shared__ double Bs[BK][BN];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int numThreads = blockDim.x * blockDim.y;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int localRow = ty * TM;
    const int localCol = tx * TN;
    double acc[TM][TN];

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    /*
        We can safely use double2 vectorized global loads when:
        - K is even, so A rows are 16-byte aligned for even k
        - N is even, so B rows are 16-byte aligned for even col
        - cudaMalloc gives sufficiently aligned base pointers
    */
    const bool canVectorize = ((K & 1) == 0) && ((N & 1) == 0);

    for (int k0 = 0; k0 < K; k0 += BK) {
        constexpr int A_VEC_ELEMS = BM * BK / 2;

        for (int idx = tid; idx < A_VEC_ELEMS; idx += numThreads) {
            int elem = idx * 2;
            int r = elem / BK;
            int k = elem - r * BK;
            int globalRow = blockRow + r;
            int globalK   = k0 + k;

            if (canVectorize && globalRow < M && globalK + 1 < K) {
                double2 v = *reinterpret_cast<const double2*>(
                    &A[globalRow * K + globalK]
                );

                *reinterpret_cast<double2*>(&As[r][k]) = v;
            } else {
                double v0 = 0.0;
                double v1 = 0.0;

                if (globalRow < M && globalK < K) {
                    v0 = A[globalRow * K + globalK];
                }

                if (globalRow < M && globalK + 1 < K) {
                    v1 = A[globalRow * K + globalK + 1];
                }

                As[r][k]     = v0;
                As[r][k + 1] = v1;
            }
        }

        constexpr int B_VEC_ELEMS = BK * BN / 2;

        for (int idx = tid; idx < B_VEC_ELEMS; idx += numThreads) {
            int elem = idx * 2;

            int k = elem / BN;
            int c = elem - k * BN;

            int globalK   = k0 + k;
            int globalCol = blockCol + c;

            if (canVectorize && globalK < K && globalCol + 1 < N) {
                double2 v = *reinterpret_cast<const double2*>(
                    &B[globalK * N + globalCol]
                );

                *reinterpret_cast<double2*>(&Bs[k][c]) = v;
            } else {
                double v0 = 0.0;
                double v1 = 0.0;

                if (globalK < K && globalCol < N) {
                    v0 = B[globalK * N + globalCol];
                }

                if (globalK < K && globalCol + 1 < N) {
                    v1 = B[globalK * N + globalCol + 1];
                }

                Bs[k][c]     = v0;
                Bs[k][c + 1] = v1;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            double aFrag[TM];
            double bFrag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                aFrag[i] = As[localRow + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                bFrag[j] = Bs[k][localCol + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fma(aFrag[i], bFrag[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int globalRow = blockRow + localRow + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int globalCol = blockCol + localCol + j;

            if (globalRow < M && globalCol < N) {
                int cIdx = globalRow * N + globalCol;
                C[cIdx] = alpha * acc[i][j] + beta * C[cIdx];
            }
        }
    }
}

using namespace nvcuda;
__global__ void dgemm_v6_fp64_tensorcore_wmma(
    int M, int N, int K, double alpha, const double* __restrict__ A, const double* __restrict__ B,
    double beta, double* __restrict__ C) {

    constexpr int WMMA_M = 8;
    constexpr int WMMA_N = 8;
    constexpr int WMMA_K = 4;
    constexpr int WARP_SIZE = 32;
    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    int tilesM = M / WMMA_M;
    int tilesN = N / WMMA_N;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    int tileId = globalWarpId;
    if (tileId >= tilesM * tilesN) {
        return;
    }
    int tileM = tileId / tilesN;
    int tileN = tileId - tileM * tilesN;
    int row = tileM * WMMA_M;
    int col = tileN * WMMA_N;

    wmma::fragment<
        wmma::matrix_a,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        double,
        wmma::row_major
    > aFrag;

    wmma::fragment<
        wmma::matrix_b,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        double,
        wmma::row_major
    > bFrag;

    wmma::fragment<
        wmma::accumulator,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        double
    > accFrag;
    wmma::fill_fragment(accFrag, 0.0);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const double* aTile = A + row * K + k0;
        const double* bTile = B + k0 * N + col;

        wmma::load_matrix_sync(aFrag, aTile, K);
        wmma::load_matrix_sync(bFrag, bTile, N);
        wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
    }

    if (beta != 0.0) {
        wmma::fragment<
            wmma::accumulator,
            WMMA_M,
            WMMA_N,
            WMMA_K,
            double
        > cOldFrag;
        const double* cTileIn = C + row * N + col;
        wmma::load_matrix_sync(cOldFrag, cTileIn, N, wmma::mem_row_major);

        #pragma unroll
        for (int i = 0; i < accFrag.num_elements; ++i) {
            accFrag.x[i] = alpha * accFrag.x[i] + beta * cOldFrag.x[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < accFrag.num_elements; ++i) {
            accFrag.x[i] = alpha * accFrag.x[i];
        }
    }
    double* cTileOut = C + row * N + col;
    wmma::store_matrix_sync(cTileOut, accFrag, N, wmma::mem_row_major);
}

template <
    int BM = 64,
    int BN = 64,
    int BK = 16
>
__global__ void dgemm_v6_fp64_wmma_tiled(
    int M, int N, int K, double alpha, const double* __restrict__ A, const double* __restrict__ B,
    double beta, double* __restrict__ C) {

    constexpr int WMMA_M = 8;
    constexpr int WMMA_N = 8;
    constexpr int WMMA_K = 4;
    constexpr int WARPS_PER_BLOCK = 16;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    static_assert(BM == 64, "This kernel expects BM = 64");
    static_assert(BN == 64, "This kernel expects BN = 64");
    static_assert(BK == 16, "This kernel expects BK = 16");

    /*
        Padding helps reduce shared-memory bank conflicts.

        AS_LD and BS_LD must remain compatible with WMMA load requirements.
        For FP64 WMMA, keeping them multiples of 4 is a good rule.
    */
    constexpr int AS_LD = BK + 4;   // 20
    constexpr int BS_LD = BN + 8;   // 72
    __shared__ __align__(16) double As[BM][AS_LD];
    __shared__ __align__(16) double Bs[BK][BS_LD];
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    /*
        There are 8 WMMA row tiles and 8 WMMA col tiles in a 64 x 64 block.

        16 warps/block.

        Mapping:
            warp_m      = 0..7
            warp_n_base = 0 or 4

        Each warp computes four 8x8 C tiles:
            same M tile
            four consecutive N tiles
    */
    const int warp_m = warpId / 2;          // 0..7
    const int warp_n_group = warpId & 1;    // 0..1
    const int warp_n_base = warp_n_group * 4;
    const int cRow = warp_m * WMMA_M;

    wmma::fragment<
        wmma::accumulator,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        double
    > acc[4];

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        wmma::fill_fragment(acc[t], 0.0);
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ============================================================
        // Cooperative vectorized load of A tile into shared memory.
        // A tile shape: 64 x 16
        // ============================================================
        constexpr int A_VEC_LOADS = BM * BK / 2;  // 512 double2 loads

        for (int idx = tid; idx < A_VEC_LOADS; idx += THREADS_PER_BLOCK) {
            int linear = idx * 2;

            int r = linear / BK;
            int k = linear - r * BK;

            const double2 v = *reinterpret_cast<const double2*>(
                &A[(blockRow + r) * K + (k0 + k)]
            );

            As[r][k]     = v.x;
            As[r][k + 1] = v.y;
        }

        // ============================================================
        // Cooperative vectorized load of B tile into shared memory.
        // B tile shape: 16 x 64
        // ============================================================

        constexpr int B_VEC_LOADS = BK * BN / 2;  // 512 double2 loads

        for (int idx = tid; idx < B_VEC_LOADS; idx += THREADS_PER_BLOCK) {
            int linear = idx * 2;

            int k = linear / BN;
            int c = linear - k * BN;

            const double2 v = *reinterpret_cast<const double2*>(
                &B[(k0 + k) * N + (blockCol + c)]
            );

            Bs[k][c]     = v.x;
            Bs[k][c + 1] = v.y;
        }

        __syncthreads();

        // ============================================================
        // Tensor Core compute on this BK slice.
        // BK = 16, WMMA_K = 4, so 4 inner WMMA steps.
        // ============================================================

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<
                wmma::matrix_a,
                WMMA_M,
                WMMA_N,
                WMMA_K,
                double,
                wmma::row_major
            > aFrag;

            wmma::load_matrix_sync(
                aFrag,
                &As[cRow][kk],
                AS_LD
            );

            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int cCol = (warp_n_base + t) * WMMA_N;

                wmma::fragment<
                    wmma::matrix_b,
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    double,
                    wmma::row_major
                > bFrag;

                wmma::load_matrix_sync(
                    bFrag,
                    &Bs[kk][cCol],
                    BS_LD
                );

                wmma::mma_sync(acc[t], aFrag, bFrag, acc[t]);
            }
        }

        __syncthreads();
    }

    // ================================================================
    // alpha/beta and store.
    // ================================================================

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int cCol = (warp_n_base + t) * WMMA_N;

        double* cPtr = C + (blockRow + cRow) * N + (blockCol + cCol);

        if (beta != 0.0) {
            wmma::fragment<
                wmma::accumulator,
                WMMA_M,
                WMMA_N,
                WMMA_K,
                double
            > cOld;

            wmma::load_matrix_sync(
                cOld,
                cPtr,
                N,
                wmma::mem_row_major
            );

            #pragma unroll
            for (int i = 0; i < acc[t].num_elements; ++i) {
                acc[t].x[i] = alpha * acc[t].x[i] + beta * cOld.x[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < acc[t].num_elements; ++i) {
                acc[t].x[i] = alpha * acc[t].x[i];
            }
        }

        wmma::store_matrix_sync(
            cPtr,
            acc[t],
            N,
            wmma::mem_row_major
        );
    }
}

template <
    int BM = 64,
    int BN = 64,
    int BK = 16
>
__global__ __launch_bounds__(256)
void dgemm_v7_fp64_wmma_64x64_8warps(
    int M, int N, int K, double alpha, const double* __restrict__ A, const double* __restrict__ B,
    double beta, double* __restrict__ C) {
    constexpr int WMMA_M = 8;
    constexpr int WMMA_N = 8;
    constexpr int WMMA_K = 4;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    static_assert(BM == 64, "BM must be 64");
//static_assert(BN == 64, "BN must be 64");
    static_assert(BK == 16, "BK must be 16");
    constexpr int AS_LD = BK + 4;   // 20
    constexpr int BS_LD = BN + 8;   // 72

    __shared__ __align__(16) double As[BM][AS_LD];
    __shared__ __align__(16) double Bs[BK][BS_LD];

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    /*
        64 x 64 C block = 8 x 8 WMMA tiles.

        8 warps/block.

        Each warp computes:
            2 WMMA tiles in M
            4 WMMA tiles in N

        So each warp computes a 16 x 32 C region.
    */

    const int warp_m_group = warpId >> 1;   // 0..3
    const int warp_n_group = warpId & 1;    // 0..1

    const int warpRowBase = warp_m_group * 16;   // 0, 16, 32, 48
    const int warpColBase = warp_n_group * 32;   // 0, 32

    wmma::fragment<
        wmma::accumulator,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        double
    > acc[2][4];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile: 64 x 16
        constexpr int A_VEC_LOADS = BM * BK / 2;

        for (int idx = tid; idx < A_VEC_LOADS; idx += THREADS_PER_BLOCK) {
            int linear = idx * 2;

            int r = linear / BK;
            int k = linear - r * BK;

            const double2 v = *reinterpret_cast<const double2*>(
                &A[(blockRow + r) * K + (k0 + k)]
            );

            As[r][k]     = v.x;
            As[r][k + 1] = v.y;
        }

        // Load B tile: 16 x 64
        constexpr int B_VEC_LOADS = BK * BN / 2;

        for (int idx = tid; idx < B_VEC_LOADS; idx += THREADS_PER_BLOCK) {
            int linear = idx * 2;

            int k = linear / BN;
            int c = linear - k * BN;

            const double2 v = *reinterpret_cast<const double2*>(
                &B[(k0 + k) * N + (blockCol + c)]
            );

            Bs[k][c]     = v.x;
            Bs[k][c + 1] = v.y;
        }

        __syncthreads();

        // Compute BK slice using FP64 Tensor Cores.
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<
                wmma::matrix_a,
                WMMA_M,
                WMMA_N,
                WMMA_K,
                double,
                wmma::row_major
            > aFrag[2];

            wmma::fragment<
                wmma::matrix_b,
                WMMA_M,
                WMMA_N,
                WMMA_K,
                double,
                wmma::row_major
            > bFrag[4];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(
                    aFrag[i],
                    &As[warpRowBase + i * WMMA_M][kk],
                    AS_LD
                );
            }

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(
                    bFrag[j],
                    &Bs[kk][warpColBase + j * WMMA_N],
                    BS_LD
                );
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store C
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int cRow = warpRowBase + i * WMMA_M;
            const int cCol = warpColBase + j * WMMA_N;

            double* cPtr = C + (blockRow + cRow) * N + (blockCol + cCol);

            if (beta != 0.0) {
                wmma::fragment<
                    wmma::accumulator,
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    double
                > cOld;

                wmma::load_matrix_sync(
                    cOld,
                    cPtr,
                    N,
                    wmma::mem_row_major
                );

                #pragma unroll
                for (int e = 0; e < acc[i][j].num_elements; ++e) {
                    acc[i][j].x[e] = alpha * acc[i][j].x[e] + beta * cOld.x[e];
                }
            } else {
                #pragma unroll
                for (int e = 0; e < acc[i][j].num_elements; ++e) {
                    acc[i][j].x[e] *= alpha;
                }
            }

            wmma::store_matrix_sync(
                cPtr,
                acc[i][j],
                N,
                wmma::mem_row_major
            );
        }
    }
}

void fill_random(std::vector<double>& x, unsigned long long seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (double& v : x) {
        v = dist(rng);
    }
}

int parse_arg(char** argv, int index, int default_value)
{
    if (argv[index] == nullptr) {
        return default_value;
    }

    return std::stoi(argv[index]);
}

int main(int argc, char** argv)
{
    // Default matrix dimensions.
    //
    // A: M x K
    // B: K x N
    // C: M x N
    int M = 1024;
    int N = 1024;
    int K = 1024;

    if (argc >= 4) {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    std::cout << "Matrix Dimensions\n";
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

    fill_random(hA, 1);
    fill_random(hB, 2);
    fill_random(hC, 3);
    reset_C_cpu(M, N, hC_comparison.data());

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
    constexpr int num_repeats = 10;

    // ============================================================
    // CPU Comparison variant (datA)
    // ============================================================
    std::cout << "Variant CPU: Naive \n";

    auto start_cpu = std::chrono::high_resolution_clock::now();
    dgemm_naive_cpu(
      M, N, K, alpha, hA.data(), hB.data(), beta, hC_comparison.data());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
    double total_seconds_cpu = elapsed_cpu.count();
    double avg_seconds_cpu = total_seconds_cpu / static_cast<double>(1);
    double gflops_cpu = flops / avg_seconds_cpu / 1.0e9;
    std::cout << "-> Runtime: " << avg_seconds_cpu * 1.0e3 << "ms\n";
    std::cout << "-> Variant CPU performance: " << gflops_cpu << " GFLOP/s\n";
    std::cout << std::endl;

    // ============================================================
    // Variant 0 launch configuration
    // ============================================================
    std::cout << "Variant 0: Naive \n";

    auto start_v0 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
        dgemm_naive_unoptimized_v0<<<grid_v1, block_v1>>>(
          M, N, K, alpha, dA, dB, beta, dC);
    }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_v0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_v0 = end_v0 - start_v0;
    double total_seconds_v0 = elapsed_v0.count();
    double avg_seconds_v0 = total_seconds_v0 / static_cast<double>(num_repeats);
    double gflops_v0 = flops / avg_seconds_v0 / 1.0e9;
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant 0 faulty!" << std::endl;
    std::cout << "-> Average Runtime: " << avg_seconds_v0 * 1.0e3 << "ms\n";
    std::cout << "-> Performance: " << gflops_v0 << " GFLOP/s\n";
    std::cout << std::endl;
      

    // ============================================================
    // Variant 1 launch configuration
    // ============================================================

    std::cout << "Variant 1: Naive but with optimized access pattern\n";
    auto start_v1 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
        dgemm_naive_v1<<<grid_v1, block_v1>>>(
          M, N, K, alpha, dA, dB, beta, dC);
    }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_v1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_v1 = end_v1 - start_v1;
    double total_seconds_v1 = elapsed_v1.count();
    double avg_seconds_v1 = total_seconds_v1 / static_cast<double>(num_repeats);
    double gflops_v1 = flops / avg_seconds_v1 / 1.0e9;

    std::cout << "-> Block: " << block_v1.x << " x " << block_v1.y << "\n";
    std::cout << "-> Grid:  " << grid_v1.x  << " x " << grid_v1.y  << "\n";
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant 1 faulty!" << std::endl;
    std::cout << "-> average runtime: " << avg_seconds_v1 * 1.0e3 << "ms\n";
    std::cout << "-> performance: " << gflops_v1 << " GLOP/s\n";
    std::cout << std::endl;

    // Optionally copy result back to host
    cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost);


    // ============================================================
    // Variant 2 launch configuration
    // ============================================================
    std::cout << "Variant 2: shared-memory tiled\n";

    constexpr int TILE_V2 = 32;
    dim3 block_v2(TILE_V2, TILE_V2);
    dim3 grid_v2(
        (N + TILE_V2 - 1) / TILE_V2,
        (M + TILE_V2 - 1) / TILE_V2
    );
    auto start_v2 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
            dgemm_shared_tiled_v2<TILE_V2><<<grid_v2, block_v2>>>(
              M, N, K, alpha, dA, dB, beta, dC);
    }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_v2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_v2 = end_v2 - start_v2;
    double total_seconds_v2 = elapsed_v2.count();
    double avg_seconds_v2 = total_seconds_v2 / static_cast<double>(num_repeats);
    double gflops_v2 = flops / avg_seconds_v2 / 1.0e9;
    std::cout << "-> Block: " << block_v2.x << " x " << block_v2.y << "\n";
    std::cout << "-> Grid:  " << grid_v2.x  << " x " << grid_v2.y  << "\n";
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant 2 faulty!" << std::endl;
    std::cout << "-> average runtime: " << avg_seconds_v2 * 1.0e3 << "ms\n";
    std::cout << "-> performance: " << gflops_v2 << " GLOP/s\n";
    std::cout << std::endl;


    // ============================================================
    // Variant 3 launch configuration
    // ============================================================
    std::cout << "Variant 3: register tiled\n";
    constexpr int BM_v3 = 64;
    constexpr int BN_v3 = 64;
    constexpr int BK_v3 = 8;
    constexpr int TM_v3 = 4;
    constexpr int TN_v3 = 4;

    static_assert(BM_v3 % TM_v3 == 0, "BM must be divisible by TM");
    static_assert(BN_v3 % TN_v3 == 0, "BN must be divisible by TN");

    dim3 block_v3(BN_v3 / TN_v3, BM_v3 / TM_v3);
    dim3 grid_v3(
        (N + BN_v3 - 1) / BN_v3,
        (M + BM_v3 - 1) / BM_v3
    );
    auto start_v3 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
            dgemm_register_tiled_v3<BM_v3, BN_v3, BK_v3, TM_v3, TN_v3><<<grid_v3, block_v3>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_v3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_v3 = end_v3 - start_v3;
    double total_seconds_v3 = elapsed_v3.count();
    double avg_seconds_v3 = total_seconds_v3 / static_cast<double>(num_repeats);
    double gflops_v3 = flops / avg_seconds_v3 / 1.0e9;

    // std::cout << "-> BM = " << BM_v3 << ", BN = " << BN_v3
    //           << ", BK = " << BK_v3 << "\n";
    // std::cout << "-> TM = " << TM_v3 << ", TN = " << TN_v3 << "\n";
    std::cout << "-> Block: " << block_v3.x << " x " << block_v3.y << "\n";
    std::cout << "-> Grid:  " << grid_v3.x  << " x " << grid_v3.y  << "\n";
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant 3 faulty!" << std::endl;
    std::cout << "-> average runtime: " << avg_seconds_v3 * 1.0e3 << "ms\n";
    std::cout << "-> performance: " << gflops_v3 << " GLOP/s\n";
    std::cout << std::endl;

    // ============================================================
    // Variant 4 launch configuration
    // ============================================================
    std::cout << "Variant 4: shared register tiled with better ccnfig\n";

    constexpr int BM_v4 = 64;
    constexpr int BN_v4 = 32;
    constexpr int BK_v4 = 16;
    constexpr int TM_v4 = 4;
    constexpr int TN_v4 = 2;

    dim3 block_v4(BN_v4 / TN_v4, BM_v4 / TM_v4);
    dim3 grid_v4(
        (N + BN_v4 - 1) / BN_v4,
        (M + BM_v4 - 1) / BM_v4
    );
    auto start_v4 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
      dgemm_v4_shared_register_tiled<BM_v4, BN_v4, BK_v4, TM_v4, TN_v4>
          <<<grid_v4, block_v4>>>(
            M, N, K, alpha, dA, dB, beta, dC);
        }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_v4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_v4 = end_v4 - start_v4;
    double total_seconds_v4 = elapsed_v4.count();
    double avg_seconds_v4 = total_seconds_v4 / static_cast<double>(num_repeats);
    double gflops_v4 = flops / avg_seconds_v4 / 1.0e9;

    // std::cout << "-> BM = " << BM_v4 << ", BN = " << BN_v4
    //           << ", BK = " << BK_v4 << "\n";
    // std::cout << "-> TM = " << TM_v4 << ", TN = " << TN_v4 << "\n";
    std::cout << "-> Block: " << block_v4.x << " x " << block_v4.y << "\n";
    std::cout << "-> Grid:  " << grid_v4.x  << " x " << grid_v4.y  << "\n";
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant 4 faulty!" << std::endl;
    std::cout << "-> average runtime: " << avg_seconds_v4 * 1.0e3 << "ms\n";
    std::cout << "-> performance: " << gflops_v4 << " GLOP/s\n";
    std::cout << std::endl;

    // ============================================================
    // Variant 5 launch configuration
    // ============================================================
    std::cout << "Variant 5: vectorized shared register tiled\n";

    constexpr int BM_v5 = 64;
    constexpr int BN_v5 = 64;
    constexpr int BK_v5 = 8;
    constexpr int TM_v5 = 4;
    constexpr int TN_v5 = 4;

    dim3 block_v5(BN_v5 / TN_v5, BM_v5 / TM_v5);
    dim3 grid_v5(
        (N + BN_v5 - 1) / BN_v5,
        (M + BM_v5 - 1) / BM_v5
    );
    auto start_v5 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
      dgemm_v5_vectorized_register_tiled<BM_v5, BN_v5, BK_v5, TM_v5, TN_v5>
          <<<grid_v5, block_v5>>>(
            M, N, K, alpha, dA, dB, beta, dC);
        }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_v5 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_v5 = end_v5 - start_v5;
    double total_seconds_v5 = elapsed_v5.count();
    double avg_seconds_v5 = total_seconds_v5 / static_cast<double>(num_repeats);
    double gflops_v5 = flops / avg_seconds_v5 / 1.0e9;

    // std::cout << "-> BM = " << BM_v5 << ", BN = " << BN_v5
    //           << ", BK = " << BK_v5 << "\n";
    // std::cout << "-> TM = " << TM_v5 << ", TN = " << TN_v5 << "\n";
    std::cout << "-> Block: " << block_v5.x << " x " << block_v5.y << "\n";
    std::cout << "-> Grid:  " << grid_v5.x  << " x " << grid_v5.y  << "\n";
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant 5 faulty!" << std::endl;
    std::cout << "-> average runtime: " << avg_seconds_v5 * 1.0e3 << "ms\n";
    std::cout << "-> performance: " << gflops_v5 << " GLOP/s\n";
    std::cout << std::endl;

    // ============================================================
    // Variant 6 launch configuration
    // ============================================================
    std::cout << "Variant 6: WMMA\n";
    
    constexpr int BM_v6 = 64;
    constexpr int BN_v6 = 64;
    constexpr int BK_v6 = 16;

    if ((M % BM_v6) != 0 || (N % BN_v6) != 0 || (K % BK_v6) != 0) {
        fprintf(stderr,
                "dgemm_v5b_fp64_wmma_tiled requires "
                "M %% 64 == 0, N %% 64 == 0, K %% 16 == 0\n");
        return 0;
    }

    dim3 block_v6(512);
    dim3 grid_v6(N / BN_v6, M / BM_v6);

    auto start_v6 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
      dgemm_v6_fp64_wmma_tiled<<<grid_v6,  block_v6>>>(
        M, N, K, alpha, dA, dB, beta, dC);
    }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_v6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_v6 = end_v6 - start_v6;
    double total_seconds_v6 = elapsed_v6.count();
    double avg_seconds_v6 = total_seconds_v6 / static_cast<double>(num_repeats);
    double gflops_v6 = flops / avg_seconds_v6 / 1.0e9;
    std::cout << "-> Block: " << block_v6.x << " x " << block_v6.y << "\n";
    std::cout << "-> Grid:  " << grid_v6.x  << " x " << grid_v6.y  << "\n";
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant 6 faulty!" << std::endl;
    std::cout << "-> average runtime: " << avg_seconds_v6 * 1.0e3 << "ms\n";
    std::cout << "-> performance: " << gflops_v6 << " GLOP/s\n";
    std::cout << std::endl;

    // ============================================================
    // Variant 7 launch configuration
    // ============================================================
    std::cout << "Variant 7: Better WMMA\n";
    constexpr int BM_v7 = 64;
    constexpr int BN_v7 = 64;
    constexpr int BK_v7 = 16;

    if ((M % BM_v7) != 0 || (N % BN_v7) != 0 || (K % BK_v7) != 0) {
        fprintf(stderr,
                "dgemm_v7_fp64_wmma_64x64_8warps requires "
                "M %% 64 == 0, N %% 64 == 0, K %% 16 == 0\n");
        return 0;
    }

    dim3 block_v7(256);
    dim3 grid_v7(N / BN_v7, M / BM_v7);

    auto start_v7 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
    dgemm_v7_fp64_wmma_64x64_8warps<BM_v7, BN_v7, BK_v7>
        <<<grid_v7, block_v7>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_v7 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_v7 = end_v7 - start_v7;
    double total_seconds_v7 = elapsed_v7.count();
    double avg_seconds_v7 = total_seconds_v7 / static_cast<double>(num_repeats);
    double gflops_v7 = flops / avg_seconds_v7 / 1.0e9;
    std::cout << "Block: " << block_v7.x << " x " << block_v7.y << "\n";
    std::cout << "Grid:  " << grid_v7.x  << " x " << grid_v7.y  << "\n";
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant 7 faulty!" << std::endl;
    std::cout << "-> average runtime: " << avg_seconds_v7 * 1.0e3 << "ms\n";
    std::cout << "-> performance: " << gflops_v7 << " GLOP/s\n";
    std::cout << std::endl;


    std::cout << "Variant CUBLAS: \n";
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    // Optional if you use a specific stream:
    // cudaStream_t stream;
    // CHECK_CUDA(cudaStreamCreate(&stream));
    // CHECK_CUBLAS(cublasSetStream(handle, stream));
    //cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    // warmup
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto start_cublas = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
    // A in cuBLAS sense: B_row interpreted as B^T
    // B in cuBLAS sense: A_row interpreted as A^T
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    }
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto end_cublas = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cublas = end_cublas - start_cublas;
    double total_seconds_cublas = elapsed_cublas.count();
    double avg_seconds_cublas = total_seconds_cublas / static_cast<double>(num_repeats);
    double gflops_cublas = flops / avg_seconds_cublas / 1.0e9;
    std::cout << "Variant CUBLAS: \n";
    if (check_gpu_results(M, N, dC, hC_comparison.data())) 
        std::cout << "-> Results correct!" << std::endl;
    else
        std::cout << "-> ERROR: Variant CUBLAS faulty!" << std::endl;
    std::cout << "-> average runtime: " << avg_seconds_cublas * 1.0e3 << "ms\n";
    std::cout << "-> performance: " << gflops_cublas << " GLOP/s\n";
    std::cout << std::endl;


    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
