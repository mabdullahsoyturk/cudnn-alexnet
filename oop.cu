#include "ConvolutionLayer.h"

int main() {
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    // Convolution Layer 1
    ConvolutionLayer convolution1(&cudnn);
    convolution1.SetInputDescriptor(1, 3, 224, 224);
    convolution1.SetFilterDescriptor(96, 3, 11, 11);
    convolution1.SetConvolutionDescriptor(2, 2, 4, 4, 1, 1);
    convolution1.SetOutputDescriptor();
    convolution1.SetAlgorithm();
    convolution1.AllocateWorkspace();
    convolution1.AllocateMemory();
    convolution1.Forward();

    // Pooling Layer 2
}