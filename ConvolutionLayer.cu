#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(cudnnHandle_t* handle): handle(handle) {}

void ConvolutionLayer::SetInputDescriptor(int N, int C, int H, int W) {
    input_n = N;
    input_c = C;
    input_h = H;
    input_w = W;

    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor, 
                                        CUDNN_TENSOR_NCHW, 
                                        CUDNN_DATA_FLOAT,
                                        input_n, input_c, input_h, input_w));
    
    printf("Input Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", input_n, input_c, input_h, input_w);
}

void ConvolutionLayer::SetFilterDescriptor(int N, int C, int H, int W) {
    filter_n = N;
    filter_c = C;
    filter_h = H;
    filter_w = W;

    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor, 
                                    CUDNN_DATA_FLOAT, 
                                    CUDNN_TENSOR_NCHW,
                                    filter_n, filter_c, filter_h, filter_w));

    printf("Filter Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", filter_n, filter_c, filter_h, filter_w);
}

void ConvolutionLayer::SetOutputDescriptor() {
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, 
                                                   input_descriptor, filter_descriptor,
                                                   &output_n, &output_c, &output_h, &output_w));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor, 
                                            CUDNN_TENSOR_NCHW, 
                                            CUDNN_DATA_FLOAT,
                                            output_n, output_c, output_h, output_w));

    printf("Convolution Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", output_n, output_c, output_h, output_w);
}

void ConvolutionLayer::SetConvolutionDescriptor(int H_padding, int W_padding, int H_stride, int W_stride, int H_dilation, int W_dilation) {
    padding_h = H_padding;
    padding_w = W_padding;
    stride_h = H_stride;
    stride_w = W_stride;
    dilation_h = H_dilation;
    dilation_w = W_dilation;

    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                                             CUDNN_CONVOLUTION, 
                                             CUDNN_DATA_FLOAT));

    printf("Convolution parameters => Padding h: %d, Padding w: %d, Stride h: %d, Stride w: %d, Dilation h: %d, Dilation w: %d\n",
                                    padding_h,     padding_w,     stride_h,     stride_w,     dilation_h,     dilation_w);
}

void ConvolutionLayer::SetAlgorithm() {
    cudnnConvolutionFwdAlgoPerf_t convolution_algo_perf;
    int algo_count;

    cudnnGetConvolutionForwardAlgorithm_v7(*handle,
                                            input_descriptor,
                                            filter_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            /*requested algo count*/1,
                                            /*returned algo count*/&algo_count,
                                            &convolution_algo_perf);
    
    algorithm = convolution_algo_perf.algo;
}

void ConvolutionLayer::AllocateMemory() {
    CUDA_CALL(cudaMalloc(&input_data, input_n * input_c * input_h * input_w * sizeof(float)));
    CUDA_CALL(cudaMalloc(&filter_data, filter_n * filter_c * filter_h * filter_w * sizeof(float)));
    CUDA_CALL(cudaMalloc(&output_data, output_n * output_c * output_h * output_w * sizeof(float)));

    fill_constant<<<input_w * input_h, input_n * input_c>>>(input_data, 1.f);
    fill_constant<<<filter_w * filter_h, filter_n * filter_c>>>(filter_data, 1.f);
    cudaDeviceSynchronize();
}

void ConvolutionLayer::AllocateWorkspace() {
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(*handle, 
                                                     input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 
                                                     algorithm, 
                                                     &workspace_size));

    CUDA_CALL(cudaMalloc(&workspace_data, workspace_size));
    printf("Workspace allocated: %ld bytes\n", workspace_size);
}

void ConvolutionLayer::Forward() {
    CUDNN_CALL(cudnnConvolutionForward(*handle,
                                     &alpha, 
                                     input_descriptor, input_data, 
                                     filter_descriptor, filter_data,
                                     convolution_descriptor, algorithm, workspace_data, workspace_size,
                                     &beta, 
                                     output_descriptor, output_data));
}