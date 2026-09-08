
void dgemm_naive_cpu(int M, int N, int K, double alpha, const double* A,
                     const double* B, double beta, double* C) {
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
void reset_C_gpu(int M, int N, double *C) {
  cudaMemset(C, 0, M * N * sizeof(double));
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
