CXX := nvcc
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_75 -std=c++11 -DDEBUG=0

all: oop

oop: oop.cu ConvolutionLayer.cu PoolingLayer.cu RELU.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) oop.cu ConvolutionLayer.cu PoolingLayer.cu RELU.cu -o oop -lcudnn

.phony: clean

clean:
	rm oop || echo -n ""
