CXX := nvcc
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include -I ./include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_75 -std=c++11 -DDEBUG=0

all: alexnet

alexnet: src/Alexnet.cu src/ConvolutionLayer.cu src/PoolingLayer.cu src/RELU.cu src/Utils.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) src/Alexnet.cu src/ConvolutionLayer.cu src/PoolingLayer.cu src/RELU.cu src/Utils.cu -o alexnet -lcudnn

.phony: clean

clean:
	rm alexnet || echo -n ""
