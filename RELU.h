#pragma once
#include "utils.h"
#include <cudnn.h>

class RELU {
    public:
        cudnnHandle_t handle;

        cudnnActivationDescriptor_t activation_descriptor;
        cudnnTensorDescriptor_t input_descriptor;

        const float alpha = 1.f;
        const float beta = 0.f;
        
        float* data;

        RELU(cudnnHandle_t handle, cudnnTensorDescriptor_t descriptor, float* data);

        void Forward();
};
