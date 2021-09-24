#include <stdio.h>
#include "utils.h"
#include <cudnn.h>

int main() {
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // Input
  const int input_n = 1;
  const int input_c = 3;
  const int input_h = 224;
  const int input_w = 224;
  printf("Input Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", input_n, input_c, input_h, input_w);

  cudnnTensorDescriptor_t input_descriptor;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor, 
                                        CUDNN_TENSOR_NCHW, 
                                        CUDNN_DATA_FLOAT,
                                        input_n, input_c, input_h, input_w));

  float *input_data;
  CUDA_CALL(cudaMalloc(&input_data, input_n * input_c * input_h * input_w * sizeof(float)));
  fill_constant<<<input_w * input_h, input_n * input_c>>>(input_data, 1.f);
  cudaDeviceSynchronize();

  // Filter
  const int filter_n = 96;
  const int filter_c = 3;
  const int filter_h = 11;
  const int filter_w = 11;
  printf("Filter Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", filter_n, filter_c, filter_h, filter_w);

  cudnnFilterDescriptor_t filter_descriptor;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor, 
                                        CUDNN_DATA_FLOAT, 
                                        CUDNN_TENSOR_NCHW,
                                        filter_n, filter_c, filter_h, filter_w));

  float *filter_data;
  CUDA_CALL(cudaMalloc(&filter_data, filter_n * filter_c * filter_h * filter_w * sizeof(float)));
  fill_constant<<<filter_w * filter_h, filter_n * filter_c>>>(filter_data, 1.f);
  cudaDeviceSynchronize();

  // Convolution
  const int padding_h = 2;
  const int padding_w = 2;
  const int stride_h = 4;
  const int stride_w = 4;
  const int dilation_h = 1;
  const int dilation_w = 1;
  printf("Convolution parameters => Padding h: %d, Padding w: %d, Stride h: %d, Stride w: %d, Dilation h: %d, Dilation w: %d\n",
                                    padding_h,     padding_w,     stride_h,     stride_w,     dilation_h,     dilation_w);

  cudnnConvolutionDescriptor_t convolution_descriptor;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                                             CUDNN_CONVOLUTION, 
                                             CUDNN_DATA_FLOAT));

  // Output
  int output_n, output_c, output_h, output_w;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, 
                                                   input_descriptor, filter_descriptor,
                                                   &output_n, &output_c, &output_h, &output_w));

  printf("Convolution 1 Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", output_n, output_c, output_h, output_w);

  cudnnTensorDescriptor_t output_descriptor;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor, 
                                        CUDNN_TENSOR_NCHW, 
                                        CUDNN_DATA_FLOAT,
                                        output_n, output_c, output_h, output_w));

  float *output_data;
  CUDA_CALL(cudaMalloc(&output_data, output_n * output_c * output_h * output_w * sizeof(float)));

  // Get the best algorithmThe first layer of AlexNet was a convolutional layer that accepted a (224×224×3) image tensor as its input. It performed a convolution operation using 96 (11×11) kernels with a stride of four and a padding of two. This produced a (55×55×96) output tensor that was then passed through a ReLu activation function then on to the next layer. The layer contained 34,944 trainable parameters.

  cudnnConvolutionFwdAlgoPerf_t convolution_algo_perf;
  int algo_count;

  cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
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
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, 
                                                     input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 
                                                     algorithm, 
                                                     &workspace_size));

  float *workspace_data;
  CUDA_CALL(cudaMalloc(&workspace_data, workspace_size));
  printf("Workspace allocated: %ld bytes\n", workspace_size);
  
  // Perform Convolution 1: 96 filters of 11x11x3 + 4 stride, padding: 2, output shape: 54x54x96
  float alpha = 1.f;
  float beta = 0.f;
  CUDNN_CALL(cudnnConvolutionForward(cudnn,
                                     &alpha, 
                                     input_descriptor, input_data, 
                                     filter_descriptor, filter_data,
                                     convolution_descriptor, algorithm, workspace_data, workspace_size,
                                     &beta, 
                                     output_descriptor, output_data));

  // Perform Pooling 1: 3x3 max pooling + 2 stride, output shape: 27x27x96
  int window_height = 3;
  int window_width = 3;
  int pooling_stride_vertical = 2;
  int pooling_stride_horizontal = 2;

  cudnnPoolingDescriptor_t pooling_descriptor;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_descriptor));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_descriptor,
                              CUDNN_POOLING_MAX,
                              CUDNN_NOT_PROPAGATE_NAN,
                              window_height,
                              window_width,
                              /*Pad H*/0,
                              /*Pad W*/0,
                              pooling_stride_vertical,
                              pooling_stride_horizontal));
  
  int pooling_output_n, pooling_output_c, pooling_output_h, pooling_output_w;
  CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(pooling_descriptor, 
                                               output_descriptor,
                                               &pooling_output_n, &pooling_output_c, &pooling_output_h, &pooling_output_w));

  printf("Pooling 1_Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", pooling_output_n, pooling_output_c, pooling_output_h, pooling_output_w);

  cudnnTensorDescriptor_t pooling_output_descriptor;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&pooling_output_descriptor));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(pooling_output_descriptor, 
                                        CUDNN_TENSOR_NCHW, 
                                        CUDNN_DATA_FLOAT,
                                        pooling_output_n, pooling_output_c, pooling_output_h, pooling_output_w));

  float *pooling_output_data;
  // Perform Convolution 2: 256 filters of 5x5x3 + 1 stride, padding: 2, output shape: 27x27x256

  // Perform Pooling 2: 3x3 max pooling + 2 stride, output shape: 13x13x256

  // Perform Convolution 3: 384 filters of 13x13x3 + 1 stride, padding: 1, output shape: 13x13x384
  // Perform Convolution 4: 384 filters of 13x13x3 + 1 stride, padding: 1, output shape: 13x13x384
  // Perform Convolution 5: 256 filters of 3x3x3 + 1 stride, padding: 1, output shape: 13x13x256
  // Perform Pooling 3: 3x3 max pooling + 2 stride, output shape: 6x6x256

  // Results
 /*  std::cout << "input_data:" << std::endl;
  print(input_data, input_n, input_c, input_h, input_w);
  
  std::cout << "filter_data:" << std::endl;
  print(filter_data, filter_n, filter_c, filter_h, filter_w);
  
  std::cout << "output_data:" << std::endl;
  print(output_data, output_n, output_c, output_h, output_w);
 */
  // Clean your mess
  CUDA_CALL(cudaFree(workspace_data));
  CUDA_CALL(cudaFree(output_data));
  CUDA_CALL(cudaFree(pooling_output_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
  CUDA_CALL(cudaFree(filter_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
  CUDA_CALL(cudaFree(input_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
  CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}
