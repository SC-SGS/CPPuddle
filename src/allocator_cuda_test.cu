#include <hpx/hpx_main.hpp> // we don't need an hpx_main that way?
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include "../include/cuda_helper.hpp"
#include "../include/buffer_manager.hpp"
#include "../include/cuda_util.hpp"
#include <cstdio>
#include <typeinfo>
#include <chrono>
#include <random>


constexpr size_t N = 20000000;
// constexpr size_t chunksize = 20000 ;
constexpr size_t chunksize = 200000 ;
constexpr size_t passes = 100;

template<class T, size_t N>
class cuda_channel {
  private:
    std::vector<T, recycle_allocator_cuda_host<T>> host_side_buffer; 
  public:
  cuda_device_buffer<T> device_side_buffer;
  cuda_channel(void) : host_side_buffer(N), device_side_buffer(N) {
  }
  void cp_from_device(cuda_helper &interface) {
   interface.copy_async(host_side_buffer.data(), device_side_buffer.device_side_buffer, 
    N * sizeof(T), cudaMemcpyDeviceToHost);
  }
  void  cp_to_device(cuda_helper &interface) {
   interface.copy_async(device_side_buffer.device_side_buffer, host_side_buffer.data(),
    N * sizeof(T), cudaMemcpyHostToDevice);
  }

  const T& operator [] (size_t index) const {return host_side_buffer[index];}
  T& operator [] (size_t index) {return host_side_buffer[index];}
};

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

__global__
void mv(const size_t startindex,const  size_t chunksize,const  size_t N,
   double *y, double *erg) {
  // Matrix is row major
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < chunksize) {
    double tmp = 0.0;
    int start_row = (startindex + i) * N;
    for (size_t i = 0; i < N; i ++) {
      double a = (i + start_row) % 2;
      tmp += a * y[i];
    }
    erg[i] = tmp;
  }
}

// #pragma nv_exec_check_disable
int main(int argc, char *argv[])
{
  thread_local cuda_helper cuda_interface(0); //one stream per HPX thread

  // Generate Problem: Repeated (unpotizmized) Matrix-Vec
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  std::vector<double> mult_vector(N + chunksize);
  std::vector<double> erg_vector(N + chunksize);
  for (auto &entry : mult_vector) {
    entry = dis(gen);
  }

  // Nonensical function, used to simulate stuff happening on the cpu 
  auto cpu_side_function = [](auto &partial_result, size_t chunksize) {
      for (size_t i = 0; i < chunksize; i++)
        partial_result[i] *= 0.5;
  };

  // Begin iterations
  constexpr size_t number_packages = N / chunksize + static_cast<int>(N % chunksize);
  size_t problemsize = N; // interface.execute does not like constexpr so we got these
  size_t chunk = chunksize;
  std::array<hpx::future<void>, number_packages> futs;
  for (size_t i = 0; i < number_packages; i++) {
    futs[i]= hpx::make_ready_future<void>();
  }

  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t pass = 0; pass < passes; pass++) {
    // Divide into work packages - Create tasks
    size_t fut_index = 0;
    for(size_t row_index = 0; row_index < N; row_index += chunksize, fut_index++) {
    futs[fut_index] = futs[fut_index].then(
      [row_index, &erg_vector, mult_vector, &problemsize, &chunk, &cpu_side_function](hpx::future<void> &&f) {
      // Recycle communication channels
      cuda_channel<double, chunksize> input;
      cuda_channel<double, chunksize> erg;
      // Copy into input array
      for (size_t i = 0; i < chunksize; i++) {
        input[i] = mult_vector[row_index + i];
      }
      input.cp_to_device(cuda_interface);
      // Launch execution
      size_t row = row_index;
      dim3 const grid_spec((chunksize + 127)/128, 1, 1);
      dim3 const threads_per_block(128, 0, 0);
      void* args[] = {&row, &chunk, &problemsize, 
      &(input.device_side_buffer), &(erg.device_side_buffer)};
      cuda_interface.execute(reinterpret_cast<void const*>(&mv), grid_spec, threads_per_block, args, 0);
      // Copy results back
      erg.cp_from_device(cuda_interface);
      // Jump away
      auto fut = cuda_interface.get_future();
      fut.get();
      // To CPU side function
      cpu_side_function(erg, chunk);
      for (size_t i = 0; i < chunksize; i++) {
      std::cerr << row_index + i << " " << i << ";";
        erg_vector[row_index + i] = input[i];
      }
    });
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "\n==>Mults took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

}
