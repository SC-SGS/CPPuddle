#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

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

  constexpr int AS_LD = BK + 4;   // 20
  constexpr int BS_LD = BN + 8;   // 72
  __shared__ __align__(16) double As[BM][AS_LD];
  __shared__ __align__(16) double Bs[BK][BS_LD];
  const int tid = threadIdx.x;
  const int warpId = tid / 32;
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

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
