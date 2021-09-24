#pragma once
#include "utils.h"
#include <cudnn.h>

class PoolingLayer {
    cudnnHandle_t* handle;

    int window_height, window_width;
    int stride_vertical, stride_horizontal;

    cudnnPoolingDescriptor_t pooling_descriptor;
    cudnnTensorDescriptor_t pooling_output_descriptor;

    float *output_data;
    
    PoolingLayer(cudnnHandle_t* handle);
}