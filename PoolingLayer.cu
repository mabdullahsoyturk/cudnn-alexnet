#include "PoolingLayer.h"

PoolingLayer::PoolingLayer(cudnnHandle_t handle): handle(handle) {
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
}

void PoolingLayer::SetInputDescriptor(cudnnTensorDescriptor_t prev_output_descriptor) {
    input_descriptor = prev_output_descriptor;
}

void PoolingLayer::SetInputData(float* data) {
    input_data = data;
}

void PoolingLayer::SetPoolingDescriptor(int window_H, int window_W, int stride_V, int stride_H) {
    window_height = window_H;
    window_width = window_W;
    stride_vertical = stride_V;
    stride_horizontal = stride_H;

    CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_descriptor,
                                           CUDNN_POOLING_MAX,
                                           CUDNN_NOT_PROPAGATE_NAN,
                                           window_height,
                                           window_width,
                                           /*Pad H*/0,
                                           /*Pad W*/0,
                                           stride_vertical,
                                           stride_horizontal));
}

void PoolingLayer::SetOutputDescriptor(int N, int C, int H, int W) {
    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(pooling_descriptor, 
                                                 input_descriptor,
                                                 &output_n, &output_c, &output_h, &output_w));

    printf("Pooling 1_Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", output_n, output_c, output_h, output_w);

    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor, 
                                          CUDNN_TENSOR_NCHW, 
                                          CUDNN_DATA_FLOAT,
                                          output_n, output_c, output_h, output_w));
}

float* PoolingLayer::GetOutputData() {
    return output_data;
}
 
void PoolingLayer::AllocateMemory() {
    CUDA_CALL(cudaMalloc(&output_data, output_n * output_c * output_h * output_w * sizeof(float)));
}

void PoolingLayer::Forward() {
    CUDNN_CALL(cudnnPoolingForward(handle,
                                   pooling_descriptor,
                                   &alpha,
                                   input_descriptor, input_data,
                                   &beta,
                                   output_descriptor, output_data));

    //cudaFree(input_data);
}