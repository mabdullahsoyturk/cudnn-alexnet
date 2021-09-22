#include <stdio.h>
#include "utils.h"
#include <cudnn.h>

int main() {
  // Init cuDNN
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // Input
  const int input_n = 1;
  const int input_c = 1;
  const int input_h = 5;
  const int input_w = 5;
  printf("Input Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", input_n, input_c, input_h, input_w);

  cudnnTensorDescriptor_t input_descriptor;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
                                        input_descriptor, 
                                        CUDNN_TENSOR_NCHW, 
                                        CUDNN_DATA_FLOAT,
                                        input_n, input_c, input_h, input_w));

  float *input_data;
  CUDA_CALL(cudaMalloc(&input_data, input_n * input_c * input_h * input_w * sizeof(float)));

  // Filter
  const int filter_n = 1;
  const int filter_c = 1;
  const int filter_h = 2;
  const int filter_w = 2;
  printf("Filter Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", filter_n, filter_c, filter_h, filter_w);

  cudnnFilterDescriptor_t filter_descriptor;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
                                        filter_descriptor, 
                                        CUDNN_DATA_FLOAT, 
                                        CUDNN_TENSOR_NCHW,
                                        filter_n, filter_c, filter_h, filter_w));

  float *filter_data;
  CUDA_CALL(cudaMalloc(&filter_data, filter_n * filter_c * filter_h * filter_w * sizeof(float)));

  // Convolution
  const int padding_h = 1;
  const int padding_w = 1;
  const int stride_h = 1;
  const int stride_w = 1;
  const int dilation_h = 1;
  const int dilation_w = 1;
  printf("Convolution parameters => Padding h: %d, Padding w: %d, Stride h: %d, Stride w: %d, Dilation h: %d, Dilation w: %d\n",
                                    padding_h,     padding_w,     stride_h,     stride_w,     dilation_h,     dilation_w);

  cudnnConvolutionDescriptor_t convolution_descriptor;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
                                             convolution_descriptor,
                                             padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                                             CUDNN_CONVOLUTION, 
                                             CUDNN_DATA_FLOAT));

  // Output
  int output_n, output_c, output_h, output_w;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
                                                   convolution_descriptor, 
                                                   input_descriptor, filter_descriptor,
                                                   &output_n, &output_c, &output_h, &output_w));

  printf("Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", output_n, output_c, output_h, output_w);

  cudnnTensorDescriptor_t output_descriptor;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
                                        output_descriptor, 
                                        CUDNN_TENSOR_NCHW, 
                                        CUDNN_DATA_FLOAT,
                                        output_n, output_c, output_h, output_w));

  float *output_data;
  CUDA_CALL(cudaMalloc(&output_data, output_n * output_c * output_h * output_w * sizeof(float)));

  // Get the best algorithm
  cudnnConvolutionFwdAlgoPerf_t convolution_algo_perf;
  int algo_count;

  cudnnGetConvolutionForwardAlgorithm_v7(
                                         cudnn,
                                         input_descriptor,
                                         filter_descriptor,
                                         convolution_descriptor,
                                         output_descriptor,
                                         /*requested algo count*/1,
                                         /*returned algo count*/&algo_count,
                                         &convolution_algo_perf);
  
  cudnnConvolutionFwdAlgo_t algorithm = convolution_algo_perf.algo;

  // Calculate how much workspace we need (Note: Not every algorithm needs workspace. If returns 0, that's okay)
  size_t workspace_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
                                                     cudnn, 
                                                     input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 
                                                     algorithm, 
                                                     &workspace_size));

  float *workspace_data;
  CUDA_CALL(cudaMalloc(&workspace_data, workspace_size));

  printf("Workspace allocated: %ld bytes\n", workspace_size);

  // Fill input and filter data
  fill_thread_id<<<input_w * input_h, input_n * input_c>>>(input_data);
  fill_constant<<<filter_w * filter_h, filter_n * filter_c>>>(filter_data, 1.f);
  cudaDeviceSynchronize();
  
  // Perform convolution
  float alpha = 1.f;
  float beta = 0.f;
  CUDNN_CALL(cudnnConvolutionForward(
                                     cudnn,
                                     &alpha, 
                                     input_descriptor, input_data, 
                                     filter_descriptor, filter_data,
                                     convolution_descriptor, algorithm, workspace_data, workspace_size,
                                     &beta, 
                                     output_descriptor, output_data));

  // Results
  std::cout << "input_data:" << std::endl;
  print(input_data, input_n, input_c, input_h, input_w);
  
  std::cout << "filter_data:" << std::endl;
  print(filter_data, filter_n, filter_c, filter_h, filter_w);
  
  std::cout << "output_data:" << std::endl;
  print(output_data, output_n, output_c, output_h, output_w);

  // Clean your mess
  CUDA_CALL(cudaFree(workspace_data));
  CUDA_CALL(cudaFree(output_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
  CUDA_CALL(cudaFree(filter_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
  CUDA_CALL(cudaFree(input_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
  CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}
