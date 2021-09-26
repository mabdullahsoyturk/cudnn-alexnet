#pragma once
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#define CUDNN_CALL(func) {                                                                         \
  cudnnStatus_t status = (func);                                                                   \
  if (status != CUDNN_STATUS_SUCCESS) {                                                            \
    std::cerr << "Error on line " << __LINE__ << ": " << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                                                                       \
  }                                                                                                \
}

#define CUDA_CALL(func) {                                                                           \
  cudaError_t status = (func);                                                                      \
  if (status != cudaSuccess) {                                                                         \
    std::cerr << "Error on line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl;  \
    std::exit(1);                                                                                   \
  }                                                                                                 \
}

__global__ __forceinline__ void fill_constant(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

inline void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(buffer.data(), data, n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost));
  
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(10) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  
  std::cout << std::endl;
}