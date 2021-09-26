#include "RELU.h"

RELU::RELU(cudnnHandle_t handle, cudnnTensorDescriptor_t descriptor, float *input_data) : 
        handle(handle), input_descriptor(descriptor), data(input_data) {
    
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_descriptor));
    CUDNN_CALL(cudnnSetActivationDescriptor(activation_descriptor,
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN,
                                            /*RELU_coef=*/0));
}

void RELU::Forward() {
    CUDNN_CALL(cudnnActivationForward(handle,
                                      activation_descriptor,
                                      &alpha,
                                      input_descriptor,
                                      data,
                                      &beta,
                                      input_descriptor,
                                      data));
}