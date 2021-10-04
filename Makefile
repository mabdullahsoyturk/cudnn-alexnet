CXX := nvcc
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_75 -std=c++11 -DDEBUG=0

all: alexnet

alexnet: alexnet.cu ConvolutionLayer.cu PoolingLayer.cu RELU.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) alexnet.cu ConvolutionLayer.cu PoolingLayer.cu RELU.cu -o alexnet -lcudnn

.phony: clean

clean:
	rm alexnet || echo -n ""
