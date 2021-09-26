#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "RELU.h"

#define IMAGE_N 1
#define IMAGE_C 3
#define IMAGE_H 224
#define IMAGE_W 224

int main() {
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    float *input_data;
    CUDA_CALL(cudaMalloc(&input_data, IMAGE_N * IMAGE_C * IMAGE_H * IMAGE_W * sizeof(float)));
    fill_constant<<<IMAGE_W * IMAGE_H, IMAGE_N * IMAGE_C>>>(input_data, 1.f);

    // Convolution Layer 1
    ConvolutionLayer convolution1(cudnn, input_data);
    convolution1.SetInputDescriptor(1, 3, IMAGE_H, IMAGE_W);
    convolution1.SetFilterDescriptor(96, 3, 11, 11);
    convolution1.SetConvolutionDescriptor(2, 2, 4, 4, 1, 1);
    convolution1.SetOutputDescriptor();
    convolution1.SetAlgorithm();
    convolution1.AllocateWorkspace();
    convolution1.AllocateMemory();
    convolution1.Forward();

    // ReLU
    RELU relu(cudnn, convolution1.GetOutputDescriptor(), convolution1.GetOutputData());

    // Pooling Layer 2
    PoolingLayer pooling1(cudnn);
    pooling1.SetInputDescriptor(convolution1.GetOutputDescriptor());
    pooling1.SetInputData(convolution1.GetOutputData());
    pooling1.SetPoolingDescriptor(3, 3, 2, 2);
    pooling1.SetOutputDescriptor(1, 96, 27, 27);
    pooling1.AllocateMemory();
    pooling1.Forward();

    // Convolution Layer 3
    ConvolutionLayer convolution2(cudnn, pooling1.GetOutputData());
    convolution2.SetInputDescriptor(1, 96, 27, 27);
    convolution2.SetFilterDescriptor(256, 3, 5, 5);
    convolution2.SetConvolutionDescriptor(2, 2, 1, 1, 1, 1);
    convolution2.SetOutputDescriptor();
    convolution2.SetAlgorithm();
    convolution2.AllocateWorkspace();
    convolution2.AllocateMemory();
    convolution2.Forward();
}